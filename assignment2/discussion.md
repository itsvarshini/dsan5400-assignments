# Discussion Questions for Assignment 2

## Problem 1

*(B)* If you only had access to the prior probabilities, would you be more likely to guess that Kennedy or Johnson authored an unlabeled paper? Why?

*ANS:* Johnson, since a larger number of the labeled files fall under Johnson's name as opposed to Kennedy's name and that would mean that there is a higher probability (assuming that the speeches we have are considered to be the full canon of speeches by both presidents) that a given unlabeled speech belongs to Johnson.

*(E)* What are your two prior probability estimates? What is the shape of the matrix storing your likelihoods? What happens when
you vary the smoothing hyperparameter alpha?

*ANS:* The two prior probability estimates are: [0.23149114 0.76850886]. The shape of the matrix storing the likelihoods is (2, 24390). As alpha gets bigger (and closer to 1), the likelihood values get smaller.

*(F)* What are the predicted authors for each of the unlabeled works?

*ANS:* My initial output from the test() function was: [np.int64(0), np.int64(0), np.int64(1), np.int64(1), np.int64(0), np.int64(0), np.int64(0), np.int64(1), np.int64(1), np.int64(1)], i.e. [kennedy, kennedy, johnson, johnson, kennedy, kennedy, kennedy, johnson, johnson, johnson]. 

However, after applying np.log() within the NB calculation, I recieved the following final outputs: [np.int64(0), np.int64(0), np.int64(0), np.int64(1), np.int64(0), np.int64(0), np.int64(0), np.int64(1), np.int64(1), np.int64(0)], i.e. [kennedy, kennedy, kennedy, johnson, kennedy, kennedy, kennedy, johnson, johnson, kennedy]

## Problem 2

*(A)* How do your predictions in Problem 1 compare with the scikit-learn implementation of Naive Bayes?

*ANS:* The predictions from the scikit-learn implementation are as follows: [0 0 0 1 0 0 1 1 1 0], i.e. [kennedy, kennedy, kennedy, johnson, kennedy, kennedy, johnson, johnson, johnson, kennedy]. This is a better prediction since these are accurate to the actual author names attached to the filename. My attempt at approaching the NB algorithm from an under-the-hood perspective produced one incorrect prediction.

## Problem 3

*(A)* Report the accuracy and F1-score of both the Naive Bayes classifier you created and the one off-the-shelf from scikit-learn in your discussion.md file.

*ANS:* 
- For NB Classifier: Accuracy = 0.8; F1-Score = 0.75
- For scikit-learn NB: Accuracy = 0.9; F1-Score = 0.8888888888888888

*(B)* For each classifier, what do you notice from the confusion matrix?

*ANS:* 
- For NB Classifier: [[5, 0],[1, 4]]
- For scikit-learn NB: [[5, 0],[2, 3]]

Observing that the True Positives and False Negatives are common between the two models, but the False Positives and True Negative numbers are different. Specifically, in my NB classifier, the True Negative number is higher while the scikit-learn model has a higher False Positive number. 