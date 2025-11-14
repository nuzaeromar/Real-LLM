import copy
from typing import List, Optional, Union, Dict
import numpy as np
from transformers import BertConfig, BertTokenizerFast, BertForMaskedLM
import torch

from sentence_transformers import SentenceTransformer, util #ADDED

from ..classification import ClassificationAttacker, Classifier, ClassifierGoal
from ...tags import TAG_English, Tag
from ...exceptions import WordNotInDictionaryException
from ...attack_assist.substitute.word import get_default_substitute, WordSubstitute
from ...attack_assist.filter_words import get_default_filter_words

from anthro_lib import ANTHRO

import spacy #ADDED
nlp = spacy.load("en_core_web_sm") #ADDED

class Feature(object):
    def __init__(self, seq_a, label):
        self.label = label
        self.seq = seq_a
        self.final_adverse = seq_a
        self.query = 0
        self.change = 0
        self.success = 0
        self.sim = 0.0
        self.changes = []

class BERTAttacker(ClassificationAttacker):
    @property
    def TAGS(self):
        return { self.__lang_tag, Tag("get_pred", "victim"), Tag("get_prob", "victim") }

    def __init__(self, 
            mlm_path : str = 'bert-large-uncased',
            k : int = 20,
            use_bpe : bool = True,
            sim_mat : Union[None, bool, WordSubstitute] = None,
            threshold_pred_score : float = 0.3,

            semantic_threshold: float = 0.8, #ADDED
            label_words: Dict[int,List[str]] = None, #ADDED

            max_length : int = 512,
            device : Optional[torch.device] = None,
            filter_words : List[str] = None
        ):
        """
        BERT-ATTACK: Adversarial Attack Against BERT Using BERT, Linyang Li, Ruotian Ma, Qipeng Guo, Xiangyang Xue, Xipeng Qiu, EMNLP2020
        `[pdf] <https://arxiv.org/abs/2004.09984>`__
        `[code] <https://github.com/LinyangLee/BERT-Attack>`__

        Args:
            mlm_path: The path to the masked language model. **Default:** 'bert-base-uncased'
            k: The k most important words / sub-words to substitute for. **Default:** 36
            use_bpe: Whether use bpe. **Default:** `True`
            sim_mat: Whether use cosine_similarity to filter out atonyms. Keep `None` for not using a sim_mat.
            threshold_pred_score: Threshold used in substitute module. **Default:** 0.3
            max_length: The maximum length of an input sentence for bert. **Default:** 512
            device: A computing device for bert.
            filter_words: A list of words that will be preserved in the attack procesudre.

        :Classifier Capacity:
            * get_pred
            * get_prob

        
        """

        self.tokenizer_mlm = BertTokenizerFast.from_pretrained(mlm_path, do_lower_case=True)
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        config_atk = BertConfig.from_pretrained(mlm_path)
        self.mlm_model = BertForMaskedLM.from_pretrained(mlm_path, config=config_atk).to(self.device)
        self.k = k
        self.use_bpe = use_bpe
        self.threshold_pred_score = threshold_pred_score

        self.semantic_threshold = semantic_threshold  #ADDED
        self.sbert_model = SentenceTransformer("sentence-transformers/bert-large-nli-mean-tokens") #ADDED
        self.sbert_model.to(self.device) #ADDED
        self.label_words=label_words #ADDED
        self.max_length = max_length
        
        self.anthro = ANTHRO() #ADDED
        self.anthro.load('./ANTHRO_Data_V1.0') #ADDED

        self.__lang_tag = TAG_English
        if filter_words is None:
            filter_words = get_default_filter_words(self.__lang_tag)
        self.filter_words = set(filter_words)

        if sim_mat is None or sim_mat is False:
            self.use_sim_mat = False
        else:
            self.use_sim_mat = True
            if sim_mat is True:
                self.substitute = get_default_substitute(self.__lang_tag)
            else:
                self.substitute = sim_mat

    def attack(self, victim: Classifier, sentence, goal: ClassifierGoal):
        x_orig = sentence.lower()

        # return None
        tokenizer = self.tokenizer_mlm
        # MLM-process
        feature = Feature(x_orig, goal.target) #goal.target is the target label (for untargeted attacks, this is original prediction)
        words, sub_words, keys = self._tokenize(feature.seq, tokenizer) #keys is the start and end indices of each word (dividing to subwords)
        max_length = self.max_length
        # original label
        inputs = tokenizer.encode_plus(feature.seq, None, add_special_tokens=True, max_length=max_length, truncation=True)
        input_ids, _ = torch.tensor(inputs["input_ids"]), torch.tensor(inputs["token_type_ids"]) #token_type_ids: all 0s here, since there’s no second segment:None here for sentence-pair

        orig_probs = torch.Tensor(victim.get_prob([feature.seq]))
        orig_probs = orig_probs[0].squeeze()
       
        current_prob = orig_probs.max()

        sub_words = ['[CLS]'] + sub_words[:2] + sub_words[2:max_length - 2] + ['[SEP]']  

        input_ids_ = torch.tensor([tokenizer.convert_tokens_to_ids(sub_words)])
        temperature = 1.5
        word_predictions = self.mlm_model(input_ids_.to(self.device))[0].squeeze()  # seq-len(sub), vocab_size
        word_predictions = word_predictions / temperature

        word_pred_scores_all, word_predictions = torch.topk(word_predictions, self.k, -1)  # seq-len k
        word_predictions = word_predictions[1:len(sub_words) + 1, :]
        word_pred_scores_all = word_pred_scores_all[1:len(sub_words) + 1, :]

        important_scores = self.get_important_scores(words, victim, current_prob, goal.target, orig_probs)
        feature.query += int(len(words))
        list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True) #[(0, score0), (1, score1), …]
        final_words = copy.deepcopy(words)

        # most_gap = 0.0
        # candidate = None

        for top_index in list_of_index:
            if feature.change > int(0.2 * (len(words))):
                feature.success = 1  # exceed
                return None

            tgt_word = words[top_index[0]]
            if tgt_word in self.filter_words:
                continue
            if keys[top_index[0]][0] > max_length - 2: #start index of this word’s subwords is beyond the truncated length?
                continue

            substitutes = word_predictions[keys[top_index[0]][0]:keys[top_index[0]][1]]  # word_predictions[start : end ] #shape (L, k)
            word_pred_scores = word_pred_scores_all[keys[top_index[0]][0]:keys[top_index[0]][1]]

            substitutes = self.get_substitues(substitutes, tokenizer, self.mlm_model, self.use_bpe, word_pred_scores, self.threshold_pred_score)
            
            if self.use_sim_mat:
                try:
                    cfs_output = self.substitute(tgt_word)
                    cos_sim_subtitutes = [elem[0] for elem in cfs_output]
                    substitutes = list(set(substitutes) & set(cos_sim_subtitutes))
                except WordNotInDictionaryException:
                    pass
                   
            most_gap = 0.0
            candidate = None

            count = 0 #ADDED
            anthro_done = False #ADDED
    
            for substitute in substitutes:               
                if substitute == tgt_word:
                    continue  # filter out original word
                if '##' in substitute:
                    continue  # filter out sub-word

                if substitute in self.filter_words:
                    continue
                
                temp_replace = final_words.copy()   
                temp_replace[top_index[0]] = substitute
                temp_text = tokenizer.convert_tokens_to_string(temp_replace)
                inputs = tokenizer.encode_plus(temp_text, None, add_special_tokens=True, max_length=max_length, truncation=True)
                input_ids = torch.tensor(inputs["input_ids"]).unsqueeze(0).to(self.device)
                seq_len = input_ids.size(1)
                
                temp_prob = torch.Tensor(victim.get_prob([temp_text]))[0].squeeze()
                temp_label = torch.argmax(temp_prob)
                feature.query += 1

                # if goal.check(feature.final_adverse, temp_label):
                #     feature.change += 1
                #     final_words[top_index[0]] = substitute
                #     feature.changes.append([keys[top_index[0]][0], substitute, tgt_word])
                #     feature.final_adverse = temp_text
                #     feature.success = 4
                #     return feature.final_adverse

                if goal.check(feature.final_adverse, temp_label):
                    with torch.no_grad():
                        emb_orig = self.sbert_model.encode(
                            [x_orig],
                            convert_to_tensor=True
                        ).to(self.device)
                        emb_adv = self.sbert_model.encode(
                            [temp_text],
                            convert_to_tensor=True
                        ).to(self.device)
                        sim_score = util.pytorch_cos_sim(emb_orig, emb_adv).item()

                    feature.sim = sim_score
                    if sim_score >= self.semantic_threshold:
                        feature.change += 1
                        final_words[top_index[0]] = substitute
                        feature.changes.append([
                            keys[top_index[0]][0],
                            substitute,
                            tgt_word
                        ])
                        feature.final_adverse = temp_text
                        feature.success = 4
                        return feature.final_adverse     
                else:
                    label_prob = temp_prob[goal.target]
                    gap = current_prob - label_prob
                    if gap > most_gap:
                        most_gap = gap
                        candidate = substitute

                #..............ANTHRO_BEGINS................#
                count += 1
                if count == 5 and not anthro_done:
                    anthro_done = True
                    # get up to 3 ANTHRO variants
                    alt3 = list(
                        self.anthro.get_similars(
                            tgt_word,
                            level=1,
                            distance=1,
                            strict=True
                        ))[:5]
                    for alt in alt3:
                        # exactly the same test logic you use below:
                        if alt == tgt_word or '##' in alt or alt in self.filter_words:
                            continue
                        temp_replace = final_words.copy()   
                        temp_replace[top_index[0]] = alt
                        temp_text = tokenizer.convert_tokens_to_string(temp_replace)
                        inputs = tokenizer.encode_plus(temp_text, None, add_special_tokens=True, max_length=max_length, truncation=True)
                        input_ids = torch.tensor(inputs["input_ids"]).unsqueeze(0).to(self.device)
                        seq_len = input_ids.size(1)
                        
                        temp_prob = torch.Tensor(victim.get_prob([temp_text]))[0].squeeze()
                        temp_label = torch.argmax(temp_prob)
                        feature.query += 1

                        if goal.check(feature.final_adverse, temp_label):
                            with torch.no_grad():
                                emb_orig = self.sbert_model.encode(
                                    [x_orig],
                                    convert_to_tensor=True
                                ).to(self.device)
                                emb_adv = self.sbert_model.encode(
                                    [temp_text],
                                    convert_to_tensor=True
                                ).to(self.device)
                                sim_score = util.pytorch_cos_sim(emb_orig, emb_adv).item()

                            feature.sim = sim_score
                            if sim_score >= self.threshold_pred_score:
                                feature.change += 1
                                final_words[top_index[0]] = alt
                                feature.changes.append([
                                    keys[top_index[0]][0],
                                    alt,
                                    tgt_word
                                ])
                                feature.final_adverse = temp_text
                                feature.success = 4
                                return feature.final_adverse     
                        else:
                            label_prob = temp_prob[goal.target]
                            gap = current_prob - label_prob
                            if gap > most_gap:
                                most_gap = gap
                                candidate = alt

                #..............ANTHRO_ENDS................#

            if most_gap > 0:
                feature.change += 1
                feature.changes.append([keys[top_index[0]][0], candidate, tgt_word])
                current_prob = current_prob - most_gap
                final_words[top_index[0]] = candidate

            
        feature.final_adverse = (tokenizer.convert_tokens_to_string(final_words))
        feature.success = 0
        # most_gap = 0.0
        # candidate = final_words.copy()

        if self.label_words and feature.label in self.label_words:
            # a) pool words from *other* labels
            pool = []
            for lbl, words in self.label_words.items():
                if lbl != feature.label:
                    pool.extend(words)
            # dedupe, keep first occurrence
            seen = set()
            pool = [w for w in pool if not (w in seen or seen.add(w))]

            # b) score each gap by saliency (lower=safer)
            saliency = self.get_important_scores(
                final_words, victim, current_prob, goal.target, orig_probs
            )
            gaps = list(range(len(final_words) + 1))
            num_gaps  = len(gaps)
            num_to_try = 5 if num_gaps >= 5 else num_gaps
            gaps.sort(key=lambda i: saliency[min(i, len(saliency)-1)])

            # # ─── POS-based replacement ─── #
            # # 1) Group pool by their POS
            # pool_pos = {}
            # for w in pool:
            #     doc_w = nlp(w)
            #     pos   = doc_w[0].pos_        # coarse POS tag
            #     pool_pos.setdefault(pos, []).append(w)

            # # 2) Tag the original sentence words
            # sent_doc = nlp(" ".join(final_words))
            # word_pos  = [tok.pos_ for tok in sent_doc]  # same length as `words`

            # most_gap = 0.0
            # # 3) Try replacing each same-POS slot
            # for idx, _ in list_of_index:
            #     # bound on edits
            #     if feature.change > int(0.2 * len(final_words)):
            #         feature.success = 1
            #         return None

            #     tgt_pos = word_pos[idx]
            #     candidates = pool_pos.get(tgt_pos, [])

            #     for cand in candidates:
            #         if cand == final_words[idx] or '##' in cand or cand in self.filter_words:
            #             continue

            #         # mask out token at idx
            #         tokens  = final_words.copy()
            #         tokens[idx] = tokenizer.mask_token  # “[MASK]”
            #         input_ids = tokenizer.convert_tokens_to_ids(tokens)

            #         with torch.no_grad():
            #             logits = self.mlm_model(
            #                 torch.tensor([input_ids]).to(self.device)
            #             )[0]  # (1, seq_len, vocab)

            #         mask_logits = logits[0, idx]
            #         probs = torch.softmax(mask_logits, dim=-1)

            #         wid = tokenizer.convert_tokens_to_ids(cand)
            #         if wid is None or probs[wid].item() < self.threshold_pred_score:
            #             continue

            #         # build & test
            #         tokens[idx] = cand
            #         cand_text   = tokenizer.convert_tokens_to_string(tokens)
            #         temp_prob  = torch.Tensor(victim.get_prob([cand_text]))[0].squeeze()
            #         temp_label = int(torch.argmax(temp_prob))
            #         feature.query += 1

            #         if goal.check(cand_text, temp_label):
            #             with torch.no_grad():
            #                 emb_orig = self.sbert_model.encode(
            #                     [x_orig],
            #                     convert_to_tensor=True
            #                 ).to(self.device)
            #                 emb_adv = self.sbert_model.encode(
            #                     [cand_text],
            #                     convert_to_tensor=True
            #                 ).to(self.device)
            #                 sim_score = util.pytorch_cos_sim(emb_orig, emb_adv).item()
            #             feature.sim = sim_score
            #             if sim_score >= self.threshold_pred_score:
            #                 feature.change += 1
            #                 words[idx] = cand
            #                 feature.changes.append([keys[idx][0], cand, sentence])
            #                 feature.final_adverse = cand_text
            #                 feature.success = 4
            #                 return cand_text
            #         else:
            #             label_prob = temp_prob[goal.target]
            #             gap = current_prob - label_prob
            #             if gap > most_gap:
            #                 most_gap = gap
            #                 candidate = candidate

            #     if most_gap > 0:
            #         feature.change += 1
            #         feature.changes.append([keys[top_index[0]][0], candidate, tgt_word])
            #         current_prob = current_prob - most_gap
            #         final_words[top_index[0]] = candidate


            # c) try each pool word at each of the 5 safest gaps
            for w in pool:
                for gap in gaps[:num_to_try]:
                    # insert a [MASK]
                    tokens = final_words.copy()
                    tokens.insert(gap, "[MASK]")

                    # run the MLM head
                    input_ids = tokenizer.convert_tokens_to_ids(tokens)
                    with torch.no_grad():
                        logits = self.mlm_model(
                            torch.tensor([input_ids]).to(self.device)
                        )[0]  # (1, seq_len, vocab)
                    mask_idx    = tokens.index("[MASK]")
                    mask_logits = logits[0, mask_idx]
                    probs = torch.softmax(mask_logits, dim=-1)

                    # only insert if MLM gives ≥30% to w
                    wid = tokenizer.convert_tokens_to_ids(w)
                    if wid is None or probs[wid].item() < self.threshold_pred_score:
                        continue

                    # build the candidate text
                    tokens[mask_idx] = w
                    cand_text = tokenizer.convert_tokens_to_string(tokens)

                    # query the classifier
                    temp_prob  = torch.Tensor(victim.get_prob([cand_text]))[0].squeeze()
                    temp_label = int(torch.argmax(temp_prob))
                    feature.query += 1

                    # if it now flips, record success and return
                    if goal.check(feature.final_adverse, temp_label):
                        feature.change += 1
                        feature.changes.append([("insert", gap), w])
                        feature.final_adverse = cand_text
                        feature.success = 4
                        return feature.final_adverse
                    # else:
                    #     label_prob = temp_prob[goal.target]
                    #     gap = current_prob - label_prob
                    #     if gap > most_gap:
                    #         most_gap = gap
                    #         candidate = cand_text

        return None


    def _tokenize(self, seq, tokenizer):
        seq = seq.replace('\n', '').lower()
        words = seq.split(' ')

        sub_words = []
        keys = []
        index = 0
        for word in words:
            sub = tokenizer.tokenize(word)
            sub_words += sub
            keys.append([index, index + len(sub)])
            index += len(sub)

        return words, sub_words, keys

    def _get_masked(self, words):
        len_text = max(len(words), 2)
        masked_words = []
        for i in range(len_text - 1):
            masked_words.append(words[0:i] + ['[UNK]'] + words[i + 1:])
        # list of words
        return masked_words
    
    def get_important_scores(self, words, tgt_model, orig_prob, orig_label, orig_probs):
        masked_words = self._get_masked(words)
        texts = [' '.join(words) for words in masked_words]  # list of text of masked words
        leave_1_probs = torch.Tensor(tgt_model.get_prob(texts))
        leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)

        import_scores = (orig_prob
                        - leave_1_probs[:, orig_label]
                        +
                        (leave_1_probs_argmax != orig_label).float()
                        * (leave_1_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0, leave_1_probs_argmax))
                        ).data.cpu().numpy()

        return import_scores

    def get_bpe_substitues(self, substitutes, tokenizer, mlm_model):
        # substitutes L, k

        substitutes = substitutes[0:12, 0:4] # maximum BPE candidates

        # find all possible candidates 
        all_substitutes = []
        for i in range(substitutes.size(0)):
            if len(all_substitutes) == 0:
                lev_i = substitutes[i]
                all_substitutes = [[int(c)] for c in lev_i]
            else:
                lev_i = []
                for all_sub in all_substitutes:
                    for j in substitutes[i]:
                        lev_i.append(all_sub + [int(j)]) 
                all_substitutes = lev_i #list of token_ids of size 4*4*..number of lev_i

        # all substitutes  list of list of token-id (all candidates)
        c_loss = torch.nn.CrossEntropyLoss(reduction='none')
        word_list = []
        # all_substitutes = all_substitutes[:24]
        all_substitutes = torch.tensor(all_substitutes) # [ N, L ]
        all_substitutes = all_substitutes[:24].to(self.device)
        # print(substitutes.size(), all_substitutes.size())
        N, L = all_substitutes.size()
        word_predictions = mlm_model(all_substitutes)[0] # N L vocab-size
        ppl = c_loss(word_predictions.view(N*L, -1), all_substitutes.view(-1)) # [ N*L ] 
        ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1)) # N  
        _, word_list = torch.sort(ppl)
        word_list = [all_substitutes[i] for i in word_list]
        final_words = []
        for word in word_list:
            tokens = [tokenizer._convert_id_to_token(int(i)) for i in word]
            text = tokenizer.convert_tokens_to_string(tokens)
            final_words.append(text)
        return final_words

    def get_substitues(self, substitutes, tokenizer, mlm_model, use_bpe, substitutes_score=None, threshold=3.0):
        # substitues L,k
        # from this matrix to recover a word
        words = []
        sub_len, k = substitutes.size()  #unpack L*k

        if sub_len == 0:
            return words
            
        elif sub_len == 1:
            for (i,j) in zip(substitutes[0], substitutes_score[0]):
                if threshold != 0 and j < threshold:
                    break
                words.append(tokenizer._convert_id_to_token(int(i)))
        else:
            if use_bpe == 1:
                words = self.get_bpe_substitues(substitutes, tokenizer, mlm_model)
            else:
                return words
        return words
    
    def get_sim_embed(self, embed_path, sim_path):
        id2word = {}
        word2id = {}

        with open(embed_path, 'r', encoding='utf-8') as ifile:
            for line in ifile:
                word = line.split()[0]
                if word not in id2word:
                    id2word[len(id2word)] = word
                    word2id[word] = len(id2word) - 1

        cos_sim = np.load(sim_path)
        return cos_sim, word2id, id2word

