# Assamese Language Model (This README file isn't fully constructed yet)


## Dateset Type:
1. Only Assamese Language,
2. Any data. it can be story, song lyrics, nursery poem, story (can be from any class)
3. It can be Song lyrics, Folk tails, Hathor `(সাথঁৰ)`, Hadhu kotha `(সাধুকথা)`


## Dateset Example:

# I have massive Assamese Dataset nearly about **45.3T (45333004592600) Tokens​**

### It has a lots of Assamese sentances from various sources, 99.9999% cleanned

### it is in Hugging Face url: `https://huggingface.co/datasets/Ranjit89/Assamese_Language_model`

### just download the `backup_data.tar.gz` file and start using it.

### happy training....

| Topic / Dataset                              | Tokens            | Approx. Scale | Source |
|----------------------------------------------|------------------:|---------------|--------|
| Poems Dataset                                | 92.6K             | 0.0000926B    | Kaggle & Sosanko Sarmah (Contributor) |
| Song Lyrics Dataset                          | 4.5M              | 0.0045B       | Kaggle (Spotify API) |
| Story Dataset                                | 52.6B             | 52.6 Billion  | HuggingFace Dataset |
| Crawled Data                                 | 7T                | 7 Trillion    | Various Web Sources |
| CC-100 Dataset                               | 5.9M              | 0.0059B       | Common Crawl |
| Qwen3 Tokens                                 | 2B                | 2 Billion     | Kaggle |
| Kaggle News Articles Dataset                 | 49.6B             | 49.6 Billion  | Kaggle |
| IndicCorp v2 (AI4Bharat)                     | 37.8T             | 37.8 Trillion | AI4Bharat Dataset |
| Assamese Monolingual Corpus (MWire-Labs)     | 38.7B             | 38.7 Billion  | MWire-Labs |
| DailyHunt Dataset                            | 184.2B            | 184.2 Billion | Rahular Varta Dataset |
| Wikipedia Dump (2019–2025)                   | 0.2T              | 200 Billion   | Wikipedia |
|                                    |         || |
| **Total**                                    | **45.3T**         | **45.333 Trillion** | |

---
---
---
---
---

### Large file copy command: 
`rsync -ah --progress /home/ranjit/Downloads/as.txt /home/ranjit/Desktop/projects/Laguage_Model/`



### i need to remove this `‎`, it is invisible, but visible by the model `[U+200E]`





---

# Total Dataset Size

* **45.3 Trillion Tokens**
* **45,333,004,592,600 total tokens**


---
---
---
---
---

```bash

tar -cf - -P \
"/home/ranjit/Desktop/projects/Laguage_Model/Ranjit_Data/real_data/1B_assamese_Tokens_Quwn3/1B_as_tokens_unfiltered.txt" \
"/home/ranjit/Desktop/projects/Laguage_Model/Ranjit_Data/real_data/1B_assamese_Tokens_Quwn3/cleanned_1B_Quwn_tokens.txt" \
"/home/ranjit/Desktop/projects/Laguage_Model/Ranjit_Data/real_data/cc-100/cc-100_assamese_text_corpora.txt" \
"/home/ranjit/Desktop/projects/Laguage_Model/Ranjit_Data/real_data/cc-100/filtered_cc-100_assamese_text_corpora.txt" \
"/home/ranjit/Desktop/projects/Laguage_Model/Ranjit_Data/real_data/IndicCorpV2_AIBharat/as.txt" \
"/home/ranjit/Desktop/projects/Laguage_Model/Ranjit_Data/real_data/kaggel_Assamese_News_Article_Dataset/nenow_preprocessed.csv" \
"/home/ranjit/Desktop/projects/Laguage_Model/Ranjit_Data/real_data/kaggel_Assamese_News_Article_Dataset/news18_preprocessed.csv" \
"/home/ranjit/Desktop/projects/Laguage_Model/Ranjit_Data/real_data/MWire-Labs-assamese-monolingual-corpus/assamese_monolingual_sentences_final_cleaned.csv" \
"/home/ranjit/Desktop/projects/Laguage_Model/Ranjit_Data/real_data/rahular_varta_DailyHuntDataset/scrapped_text.txt" \
"/home/ranjit/Desktop/projects/Laguage_Model/Ranjit_Data/real_data/cleaned.txt" \
"/home/ranjit/Desktop/projects/Laguage_Model/Ranjit_Data/real_data/day2_data.txt" \
"/home/ranjit/Desktop/projects/Laguage_Model/Ranjit_Data/real_data/day3_day3.txt" \
"/home/ranjit/Desktop/projects/Laguage_Model/Ranjit_Data/real_data/poem.txt" \
| pv -s "$(du -cb \
"/home/ranjit/Desktop/projects/Laguage_Model/Ranjit_Data/real_data/1B_assamese_Tokens_Quwn3/1B_as_tokens_unfiltered.txt" \
"/home/ranjit/Desktop/projects/Laguage_Model/Ranjit_Data/real_data/1B_assamese_Tokens_Quwn3/cleanned_1B_Quwn_tokens.txt" \
"/home/ranjit/Desktop/projects/Laguage_Model/Ranjit_Data/real_data/cc-100/cc-100_assamese_text_corpora.txt" \
"/home/ranjit/Desktop/projects/Laguage_Model/Ranjit_Data/real_data/cc-100/filtered_cc-100_assamese_text_corpora.txt" \
"/home/ranjit/Desktop/projects/Laguage_Model/Ranjit_Data/real_data/IndicCorpV2_AIBharat/as.txt" \
"/home/ranjit/Desktop/projects/Laguage_Model/Ranjit_Data/real_data/kaggel_Assamese_News_Article_Dataset/nenow_preprocessed.csv" \
"/home/ranjit/Desktop/projects/Laguage_Model/Ranjit_Data/real_data/kaggel_Assamese_News_Article_Dataset/news18_preprocessed.csv" \
"/home/ranjit/Desktop/projects/Laguage_Model/Ranjit_Data/real_data/MWire-Labs-assamese-monolingual-corpus/assamese_monolingual_sentences_final_cleaned.csv" \
"/home/ranjit/Desktop/projects/Laguage_Model/Ranjit_Data/real_data/rahular_varta_DailyHuntDataset/scrapped_text.txt" \
"/home/ranjit/Desktop/projects/Laguage_Model/Ranjit_Data/real_data/cleaned.txt" \
"/home/ranjit/Desktop/projects/Laguage_Model/Ranjit_Data/real_data/day2_data.txt" \
"/home/ranjit/Desktop/projects/Laguage_Model/Ranjit_Data/real_data/day3_day3.txt" \
"/home/ranjit/Desktop/projects/Laguage_Model/Ranjit_Data/real_data/poem.txt" | awk '/total$/ {print $1}')" \
| gzip > backup_data.tar.gz && sync && shutdown
```

# Contant me at: 
## 📧 Ranjit: ranjitdax89@gmail.com
## 📞 Ph No: +91-9387480826