from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch 
import sys

# using totch to get the predtion from the model 
def predict_mask_sentence(sentences , model, tokenizer, dic_path):

    model.eval()
    path = dic_path + 'dictabert_results.txt'
    with open(path, 'w', encoding='utf-8') as f:     
        for sentence in sentences:

            sentence_eval = sentence.replace("[*]", tokenizer.mask_token)
    
            with torch.no_grad():
                output = model(tokenizer.encode(sentence_eval, return_tensors='pt'))

            sentece_tokenize = sentence.strip().split(' ')
            print(sentece_tokenize)
            new_sentence = []
            tokens = []
            c = 1
            for token in sentece_tokenize:
                # check if was a [*] at this place if yes take the predicted word from our model predtion 
                if token == '[*]':
                    top_1 = torch.topk(output.logits[0, c, :], 1)[1]
                    predicted_token = tokenizer.convert_ids_to_tokens(top_1)
                    tokens.append(predicted_token[0])
                    new_sentence.append(predicted_token[0])
                else:
                    predicted_token = token

                    new_sentence.append(predicted_token)
                
                c+=1


            f.write(f"Original sentence: {sentence.strip()}\n")
            f.write(f"DictaBERT sentence: {' '.join(new_sentence).strip()}\n")
            f.write(f"DictaBERT tokens: {tokens}\n\n")
   
    return 
    

if __name__=="__main__":

    if len(sys.argv) != 3:
        print("Please provode all the required arguments run file mask_sentence path and dic path")
        sys.exit(1)

    mask_path, dic_path = sys.argv[1], sys.argv[2]
    sentences = []
    with open(mask_path, 'r', encoding='utf-8') as f: 
            for line in f:
                sentences.append(line)

    model = AutoModelForMaskedLM.from_pretrained('dicta-il/dictabert')
    tokenizer = AutoTokenizer.from_pretrained('dicta-il/dictabert')

    predict_mask_sentence(sentences, model, tokenizer, dic_path)
