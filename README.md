# Assamese Conversational Language Model


## Dateset Type:
1. Only Assamese Language,
2. Any data. it can be story, song lyrics, nursery poem, story (can be from any class)
3. It can be Song lyrics, Folk tails, Hathor `(সাথঁৰ)`, Hadhu kotha `(সাধুকথা)`

##### Goal: Just collecct data
## How to contribute: 
### Steps
1. Clone the repository
2. Create a folder with your name `(eg: <your_name>_data)`
3. Inside that folder add the `txt` files as shown below
4. Now all set, just blindly add data from various sources

## Dateset Example:


### 🧩 **1. Raw Text Format ( multiple big text files `(wikipedia.txt`, `raddit_data.txt`,`news.txt`, `conversation.txt`,`story.txt`, `poem.txt`, `assamese_slang_words.txt` etc etc ):**


### Example: `(story.txt)`:
```html
    এইখন বৃষ্টিৰ পুৱাত মই একেলগে কফি খাইছিলোঁ।
    মই ভাবিলোঁ, জীৱনটো কিমান ধুনীয়া!
    তোমাৰ আজিৰ দিনটো কেনেকৈ গ’ল?
    অতি ধুনীয়া কাব্য এটা লিখিছোঁ — শুনিবা নে?

    এদিন এটা গাঁওত এজনী ছোৱালী আছিল।
    তেওঁ অতি সাহসী আৰু বুদ্ধিমতী আছিল।
    ....
    ....
```

✅ **Use case:** Masked language modeling or next-token prediction.
✅ **Easy to preprocess:** Just split into sentences and tokenize.

---

### 💬 **2. Conversation-style Format `(conversation.txt)`**

```html
    নমস্কাৰ, আপুনি কেনে আছেন?,
    মই ভাল আছোঁ, ধন্যবাদ। আপুনি কেনে আছে?,
    আজিৰ বতৰটো বৰ সুন্দৰ নহয় নে?,
    হয়, অলপ বৰষুণ আহিছে, কিন্তু বতাহটো একেবাৰে সতেজ।

    মানুহ: আপুনি কি কাম কৰে?
    সহায়ক: মই এখন AI মডেল, যি অসমীয়া ভাষাত কথা ক’ব পাৰে।
```

---

### 📚 **3. Poem `(poem.txt)`**
```html
    বতাহ আহে, মন উৰে,
    স্বপ্নৰ দিশে ওলমে চৰে।
    ....
    ....

    বতাহ আহে, মন উৰে,
    স্বপ্নৰ দিশে ওলমে চৰে।
    ....
    ....
```



### 4. Wikipedia `(wikipedia.txt)`
```html
    অসম ভাৰতৰ উত্তৰ-পূৰ্বাঞ্চলৰ এটা ৰাজ্য।
    ইতিহাস, সংস্কৃতি আৰু সংগীতত ই অত্যন্ত সমৃদ্ধ।
....
....
```

### 5. Assamese slang words `(assamese_slang_words.txt)`:
```html
<!-- This will help model to distinguish betwwen the phonetics -->
মাকচদু
বনিৰ/বুনৰী/ৰনিদ
কলা
চদুৰভাই
জহিৰী
গদা

```
# Goal: Just collect data from various sources. (Language: only Assamese)
## Your Goal: Contribute with 7GB dataset

# Contant me at: 
## 📧 Ranjit: ranjitdax89@gmail.com
## 📞 Ph No: +91-9387480826




i need to remove this ‎
`‎` it is invisible, but visible by the model
`[U+200E]`