from textblob import TextBlob

text = "Tesla stock is amazing!"
blob = TextBlob(text)
print(blob.sentiment)