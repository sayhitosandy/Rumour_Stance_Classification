# Rumour Stance Classification using Improved Feature Extraction

The task of determining veracity and authenticity of social media content has been
of recent interest to the field of NLP. False
claims and rumours affect peoples perceptions of events and their behaviour,
sometimes in harmful ways. Here, the
task is to classify tweets in threads of
tweets based on the stance possessed by
the tweet, which can be of 4 categories:
supporting (S), denying (D), querying (Q),
or commenting (C), i.e., SDQC. 

Dataset used: SemEval-2017 Task 8 Dataset: http://alt.qcri.org/semeval2017/task8/index.php?id=data-and-tools

By improving feature extraction and incorporating
tweet dependent (and also other textual) features
such as hashtags, link content etc., we were able to
achieve accuracies close to that of the state of the
art on most of our models. Our highest reported
accuracy is 77.6%, which is comparable to that of
the state of the art model (78.4%). We also tried
to improve the recall on Deny and Query classes
by augmenting the training dataset. We also tried
to do the same using ensemble methods, and bagging/boosting techniques.
