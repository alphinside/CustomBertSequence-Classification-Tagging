# How to Use

1. Download the model
2. Construct the object like example below

        from sentiment_label import BertKemenkunhamSentimentClassification

        sentiment_classifier = BertKemenkunhamSentimentClassification(
            gpu_id=0, # Use -1 for cpu
            models_dir='models', # Directory containing models,tokenizer, etc from training output
            labels_list=['netral', 'negatif', 'positif']
        )

3. Predict some text

        text = 'indexalaw rocks \n something something somesucks.'

        result = sentiment_classifier.predict(text)

4. The output is in dictionary format, with key is the string labels and value is float probability. Select the highest value as appointed label

        print(result)

        """
        {
            'netral': 0.94330186, 
            'negatif': 0.021998325, 
            'positif': 0.03469982
        }
        """
