�]q (XO�  Machine learning (ML) is the scientific study of algorithms and statistical models that computer systems use to perform a specific task without using explicit instructions, relying on patterns and inference instead. It is seen as a subset of artificial intelligence. Machine learning algorithms build a mathematical model based on sample data, known as "training data", in order to make predictions or decisions without being explicitly programmed to perform the task.[1][2]:2 Machine learning algorithms are used in a wide variety of applications, such as email filtering and computer vision, where it is difficult or infeasible to develop a conventional algorithm for effectively performing the task.
Machine learning is closely related to computational statistics, which focuses on making predictions using computers. The study of mathematical optimization delivers methods, theory and application domains to the field of machine learning. Data mining is a field of study within machine learning, and focuses on exploratory data analysis through unsupervised learning.[3][4] In its application across business problems, machine learning is also referred to as predictive analytics.
The name machine learning was coined in 1959 by Arthur Samuel.[5] Tom M. Mitchell provided a widely quoted, more formal definition of the algorithms studied in the machine learning field: "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P if its performance at tasks in T, as measured by P,  improves with experience E."[6] This definition of the tasks in which machine learning is concerned offers a fundamentally operational definition rather than defining the field in cognitive terms. This follows Alan Turing's proposal in his paper "Computing Machinery and Intelligence", in which the question "Can machines think?" is replaced with the question "Can machines do what we (as thinking entities) can do?".[7] In Turing's proposal the various characteristics that could be possessed by a thinking machine and the various implications in constructing one are exposed.

Machine learning tasks are classified into several broad categories. In supervised learning, the algorithm builds a mathematical model from a set of data that contains both the inputs and the desired outputs. For example, if the task were determining whether an image contained a certain object, the training data for a supervised learning algorithm would include images with and without that object (the input), and each image would have a label (the output) designating whether it contained the object. In special cases, the input may be only partially available, or restricted to special feedback.[clarification needed] Semi-supervised learning algorithms develop mathematical models from incomplete training data, where a portion of the sample input doesn't have labels.
Classification algorithms and regression algorithms are types of supervised learning. Classification algorithms are used when the outputs are restricted to a limited set of values. For a classification algorithm that filters emails, the input would be an incoming email, and the output would be the name of the folder in which to file the email. For an algorithm that identifies spam emails, the output would be the prediction of either "spam" or "not spam", represented by the Boolean values true and false. Regression algorithms are named for their continuous outputs, meaning they may have any value within a range. Examples of a continuous value are the temperature, length, or price of an object.
In unsupervised learning, the algorithm builds a mathematical model from a set of data that contains only inputs and no desired output labels. Unsupervised learning algorithms are used to find structure in the data, like grouping or clustering of data points. Unsupervised learning can discover patterns in the data, and can group the inputs into categories, as in feature learning. Dimensionality reduction is the process of reducing the number of "features", or inputs, in a set of data.
Active learning algorithms access the desired outputs (training labels) for a limited set of inputs based on a budget and optimize the choice of inputs for which it will acquire training labels. When used interactively, these can be presented to a human user for labeling. Reinforcement learning algorithms are given feedback in the form of positive or negative reinforcement in a dynamic environment and are used in autonomous vehicles or in learning to play a game against a human opponent.[2]:3 Other specialized algorithms in machine learning include topic modeling, where the computer program is given a set of natural language documents and finds other documents that cover similar topics. Machine learning algorithms can be used to find the unobservable probability density function in density estimation problems. Meta learning algorithms learn their own inductive bias based on previous experience. In developmental robotics, robot learning algorithms generate their own sequences of learning experiences, also known as a curriculum, to cumulatively acquire new skills through self-guided exploration and social interaction with humans. These robots use guidance mechanisms such as active learning, maturation, motor synergies, and imitation.[clarification needed]
Arthur Samuel, an American pioneer in the field of computer gaming and artificial intelligence, coined the term "Machine Learning" in 1959 while at IBM.[8] A representative book of the machine learning research during the 1960s was the Nilsson's book on Learning Machines, dealing mostly with machine learning for pattern classification.[9] The interest of machine learning related to pattern recognition continued during the 1970s, as described in the book of Duda and Hart in 1973. [10] In 1981 a report was given on using teaching strategies so that a neural network learns to recognize 40 characters (26 letters, 10 digits, and 4 special symbols) from a computer terminal. [11] 
As a scientific endeavor, machine learning grew out of the quest for artificial intelligence. Already in the early days of AI as an academic discipline, some researchers were interested in having machines learn from data. They attempted to approach the problem with various symbolic methods, as well as what were then termed "neural networks"; these were mostly perceptrons and other models that were later found to be reinventions of the generalized linear models of statistics.[12] Probabilistic reasoning was also employed, especially in automated medical diagnosis.[13]:488
However, an increasing emphasis on the logical, knowledge-based approach caused a rift between AI and machine learning. Probabilistic systems were plagued by theoretical and practical problems of data acquisition and representation.[13]:488 By 1980, expert systems had come to dominate AI, and statistics was out of favor.[14] Work on symbolic/knowledge-based learning did continue within AI, leading to inductive logic programming, but the more statistical line of research was now outside the field of AI proper, in pattern recognition and information retrieval.[13]:708–710; 755 Neural networks research had been abandoned by AI and computer science around the same time. This line, too, was continued outside the AI/CS field, as "connectionism", by researchers from other disciplines including Hopfield, Rumelhart and Hinton. Their main success came in the mid-1980s with the reinvention of backpropagation.[13]:25
Machine learning, reorganized as a separate field, started to flourish in the 1990s. The field changed its goal from achieving artificial intelligence to tackling solvable problems of a practical nature. It shifted focus away from the symbolic approaches it had inherited from AI, and toward methods and models borrowed from statistics and probability theory.[14] It also benefited from the increasing availability of digitized information, and the ability to distribute it via the Internet.
Machine learning and data mining often employ the same methods and overlap significantly, but while machine learning focuses on prediction, based on known properties learned from the training data, data mining focuses on the discovery of (previously) unknown properties in the data (this is the analysis step of knowledge discovery in databases). Data mining uses many machine learning methods, but with different goals; on the other hand, machine learning also employs data mining methods as "unsupervised learning" or as a preprocessing step to improve learner accuracy. Much of the confusion between these two research communities (which do often have separate conferences and separate journals, ECML PKDD being a major exception) comes from the basic assumptions they work with: in machine learning, performance is usually evaluated with respect to the ability to reproduce known knowledge, while in knowledge discovery and data mining (KDD) the key task is the discovery of previously unknown knowledge. Evaluated with respect to known knowledge, an uninformed (unsupervised) method will easily be outperformed by other supervised methods, while in a typical KDD task, supervised methods cannot be used due to the unavailability of training data.
Machine learning also has intimate ties to optimization: many learning problems are formulated as minimization of some loss function on a training set of examples. Loss functions express the discrepancy between the predictions of the model being trained and the actual problem instances (for example, in classification, one wants to assign a label to instances, and models are trained to correctly predict the pre-assigned labels of a set of examples). The difference between the two fields arises from the goal of generalization: while optimization algorithms can minimize the loss on a training set, machine learning is concerned with minimizing the loss on unseen samples.[15]
Machine learning and statistics are closely related fields in terms of methods, but distinct in their principal goal: statistics draws population inferences from a sample, while machine learning finds generalizable predictive patterns.[16] According to Michael I. Jordan, the ideas of machine learning, from methodological principles to theoretical tools, have had a long pre-history in statistics.[17] He also suggested the term data science as a placeholder to call the overall field.[17]
Leo Breiman distinguished two statistical modeling paradigms: data model and algorithmic model,[18] wherein "algorithmic model" means more or less the machine learning algorithms like Random forest.
Some statisticians have adopted methods from machine learning, leading to a combined field that they call statistical learning.[19]
A core objective of a learner is to generalize from its experience.[2][20] Generalization in this context is the ability of a learning machine to perform accurately on new, unseen examples/tasks after having experienced a learning data set. The training examples come from some generally unknown probability distribution (considered representative of the space of occurrences) and the learner has to build a general model about this space that enables it to produce sufficiently accurate predictions in new cases.
The computational analysis of machine learning algorithms and their performance is a branch of theoretical computer science known as computational learning theory. Because training sets are finite and the future is uncertain, learning theory usually does not yield guarantees of the performance of algorithms. Instead, probabilistic bounds on the performance are quite common. The bias–variance decomposition is one way to quantify generalization error.
For the best performance in the context of generalization, the complexity of the hypothesis should match the complexity of the function underlying the data. If the hypothesis is less complex than the function, then the model has under fitted the data. If the complexity of the model is increased in response, then the training error decreases. But if the hypothesis is too complex, then the model is subject to overfitting and generalization will be poorer.[21]
In addition to performance bounds, learning theorists study the time complexity and feasibility of learning. In computational learning theory, a computation is considered feasible if it can be done in polynomial time. There are two kinds of time complexity results. Positive results show that a certain class of functions can be learned in polynomial time. Negative results show that certain classes cannot be learned in polynomial time.
The types of machine learning algorithms differ in their approach, the type of data they input and output, and the type of task or problem that they are intended to solve.
Supervised learning algorithms build a mathematical model of a set of data that contains both the inputs and the desired outputs.[22] The data is known as training data, and consists of a set of training examples. Each training example has one or more inputs and the desired output, also known as a supervisory signal.  In the mathematical model, each training example is represented by an array or vector, sometimes called a feature vector, and the training data is represented by a matrix. Through iterative optimization of an objective function, supervised learning algorithms learn a function that can be used to predict the output associated with new inputs.[23] An optimal function will allow the algorithm to correctly determine the output for inputs that were not a part of the training data. An algorithm that improves the accuracy of its outputs or predictions over time is said to have learned to perform that task.[6]
Supervised learning algorithms include classification and regression.[24] Classification algorithms are used when the outputs are restricted to a limited set of values, and regression algorithms are used when the outputs may have any numerical value within a range. Similarity learning is an area of supervised machine learning closely related to regression and classification, but the goal is to learn from examples using a similarity function that measures how similar or related two objects are. It has applications in ranking, recommendation systems, visual identity tracking, face verification, and speaker verification.
In the case of semi-supervised learning algorithms, some of the training examples are missing training labels, but they can nevertheless be used to improve the quality of a model. In weakly supervised learning, the training labels are noisy, limited, or imprecise; however, these labels are often cheaper to obtain, resulting in larger effective training sets.[25]
Unsupervised learning algorithms take a set of data that contains only inputs, and find structure in the data, like grouping or clustering of data points. The algorithms, therefore, learn from test data that has not been labeled, classified or categorized. Instead of responding to feedback, unsupervised learning algorithms identify commonalities in the data and react based on the presence or absence of such commonalities in each new piece of data. A central application of unsupervised learning is in the field of density estimation in statistics,[26] though unsupervised learning encompasses other domains involving summarizing and explaining data features.
Cluster analysis is the assignment of a set of observations into subsets (called clusters) so that observations within the same cluster are similar according to one or more predesignated criteria, while observations drawn from different clusters are dissimilar. Different clustering techniques make different assumptions on the structure of the data, often defined by some similarity metric and evaluated, for example, by internal compactness, or the similarity between members of the same cluster, and separation, the difference between clusters. Other methods are based on estimated density and graph connectivity.
Semi-supervised learning
Semi-supervised learning falls between unsupervised learning (without any labeled training data) and supervised learning (with completely labeled training data).  Many machine-learning researchers have found that unlabeled data, when used in conjunction with a small amount of labeled data, can produce a considerable improvement in learning accuracy.
Reinforcement learning is an area of machine learning concerned with how software agents ought to take actions in an environment so as to maximize some notion of cumulative reward. Due to its generality, the field is studied in many other disciplines, such as game theory, control theory, operations research, information theory, simulation-based optimization, multi-agent systems, swarm intelligence, statistics and genetic algorithms. In machine learning, the environment is typically represented as a Markov Decision Process (MDP). Many reinforcement learning algorithms use dynamic programming techniques.[27] Reinforcement learning algorithms do not assume knowledge of an exact mathematical model of the MDP, and are used when exact models are infeasible. Reinforcement learning algorithms are used in autonomous vehicles or in learning to play a game against a human opponent.
Self-learning as machine learning paradigm was introduced in 1982 along with a neural network capable of self-learning  named Crossbar Adaptive Array (CAA). [28] It is a learning with no external rewards and no external teacher advices. The CAA self-learning algorithm computes, in a crossbar fashion, both decisions about actions and emotions (feelings) about consequence situations. The system is driven by the interaction between cognition and emotion. [29]
The self-learning algorithm updates a memory matrix W =||w(a,s)|| such that in each iteration executes the following machine learning  routine: 
It is a system with only one input, situation s, and only one output, action (or behavior) a. There is neither a separate reinforcement input nor an advice input from the environment. The backpropagated value (secondary reinforcement) is the emotion toward the consequence situation. The CAA exists in two environments, one is behavioral environment where it behaves, and the other is genetic environment, wherefrom it initially and only once receives initial emotions about situations to be encountered in the  behavioral environment. After receiving the genome (species) vector from the genetic environment, the CAA learns a goal seeking behavior, in an environment that contains both desirable and undesirable situations. [30]
Several learning algorithms aim at discovering better representations of the inputs provided during training.[31] Classic examples include principal components analysis and cluster analysis. Feature learning algorithms, also called representation learning algorithms, often attempt to preserve the information in their input but also transform it in a way that makes it useful, often as a pre-processing step before performing classification or predictions. This technique allows reconstruction of the inputs coming from the unknown data-generating distribution, while not being necessarily faithful to configurations that are implausible under that distribution. This replaces manual feature engineering, and allows a machine to both learn the features and use them to perform a specific task.
Feature learning can be either supervised or unsupervised. In supervised feature learning, features are learned using labeled input data. Examples include artificial neural networks, multilayer perceptrons, and supervised dictionary learning. In unsupervised feature learning, features are learned with unlabeled input data.  Examples include dictionary learning, independent component analysis, autoencoders, matrix factorization[32] and various forms of clustering.[33][34][35]
Manifold learning algorithms attempt to do so under the constraint that the learned representation is low-dimensional. Sparse coding algorithms attempt to do so under the constraint that the learned representation is sparse, meaning that the mathematical model has many zeros. Multilinear subspace learning algorithms aim to learn low-dimensional representations directly from tensor representations for multidimensional data, without reshaping them into higher-dimensional vectors.[36] Deep learning algorithms discover multiple levels of representation, or a hierarchy of features, with higher-level, more abstract features defined in terms of (or generating) lower-level features. It has been argued that an intelligent machine is one that learns a representation that disentangles the underlying factors of variation that explain the observed data.[37]
Feature learning is motivated by the fact that machine learning tasks such as classification often require input that is mathematically and computationally convenient to process. However, real-world data such as images, video, and sensory data has not yielded to attempts to algorithmically define specific features. An alternative is to discover such features or representations through examination, without relying on explicit algorithms.
Sparse dictionary learning is a feature learning method where a training example is represented as a linear combination of basis functions, and is assumed to be a sparse matrix. The method is strongly NP-hard and difficult to solve approximately.[38] A popular heuristic method for sparse dictionary learning is the K-SVD algorithm. Sparse dictionary learning has been applied in several contexts. In classification, the problem is to determine the class to which a previously unseen training example belongs. For a dictionary where each class has already been built, a new training example is associated with the class that is best sparsely represented by the corresponding dictionary. Sparse dictionary learning has also been applied in image de-noising. The key idea is that a clean image patch can be sparsely represented by an image dictionary, but the noise cannot.[39]
In data mining, anomaly detection, also known as outlier detection, is the identification of rare items, events or observations which raise suspicions by differing significantly from the majority of the data.[40] Typically, the anomalous items represent an issue such as bank fraud, a structural defect, medical problems or errors in a text. Anomalies are referred to as outliers, novelties, noise, deviations and exceptions.[41]
In particular, in the context of abuse and network intrusion detection, the interesting objects are often not rare objects, but unexpected bursts in activity. This pattern does not adhere to the common statistical definition of an outlier as a rare object, and many outlier detection methods (in particular, unsupervised algorithms) will fail on such data, unless it has been aggregated appropriately. Instead, a cluster analysis algorithm may be able to detect the micro-clusters formed by these patterns.[42]
Three broad categories of anomaly detection techniques exist.[43] Unsupervised anomaly detection techniques detect anomalies in an unlabeled test data set under the assumption that the majority of the instances in the data set are normal, by looking for instances that seem to fit least to the remainder of the data set. Supervised anomaly detection techniques require a data set that has been labeled as "normal" and "abnormal" and involves training a classifier (the key difference to many other statistical classification problems is the inherently unbalanced nature of outlier detection). Semi-supervised anomaly detection techniques construct a model representing normal behavior from a given normal training data set and then test the likelihood of a test instance to be generated by the model.
Association rule learning is a rule-based machine learning method for discovering relationships between variables in large databases. It is intended to identify strong rules discovered in databases using some measure of "interestingness".[44]
Rule-based machine learning is a general term for any machine learning method that identifies, learns, or evolves "rules" to store, manipulate or apply knowledge. The defining characteristic of a rule-based machine learning algorithm is the identification and utilization of a set of relational rules that collectively represent the knowledge captured by the system. This is in contrast to other machine learning algorithms that commonly identify a singular model that can be universally applied to any instance in order to make a prediction.[45] Rule-based machine learning approaches include learning classifier systems, association rule learning, and artificial immune systems.
Based on the concept of strong rules, Rakesh Agrawal, Tomasz Imieliński and Arun Swami introduced association rules for discovering regularities between products in large-scale transaction data recorded by point-of-sale (POS) systems in supermarkets.[46] For example, the rule 



{

o
n
i
o
n
s
,
p
o
t
a
t
o
e
s

}
⇒
{

b
u
r
g
e
r

}


{\displaystyle \{\mathrm {onions,potatoes} \}\Rightarrow \{\mathrm {burger} \}}

 found in the sales data of a supermarket would indicate that if a customer buys onions and potatoes together, they are likely to also buy hamburger meat. Such information can be used as the basis for decisions about marketing activities such as promotional pricing or product placements. In addition to market basket analysis, association rules are employed today in application areas including Web usage mining, intrusion detection, continuous production, and bioinformatics. In contrast with sequence mining, association rule learning typically does not consider the order of items either within a transaction or across transactions.
Learning classifier systems (LCS) are a family of rule-based machine learning algorithms that combine a discovery component, typically a genetic algorithm, with a learning component, performing either supervised learning, reinforcement learning, or unsupervised learning. They seek to identify a set of context-dependent rules that collectively store and apply knowledge in a piecewise manner in order to make predictions.[47]
Inductive logic programming (ILP) is an approach to rule-learning using logic programming as a uniform representation for input examples, background knowledge, and hypotheses. Given an encoding of the known background knowledge and a set of examples represented as a logical database of facts, an ILP system will derive a hypothesized logic program that entails all positive and no negative examples. Inductive programming is a related field that considers any kind of programming languages for representing hypotheses (and not only logic programming), such as functional programs.
Inductive logic programming is particularly useful in bioinformatics and natural language processing. Gordon Plotkin and Ehud Shapiro laid the initial theoretical foundation for inductive machine learning in a logical setting.[48][49][50] Shapiro built their first implementation (Model Inference System) in 1981: a Prolog program that inductively inferred logic programs from positive and negative examples.[51] The term inductive here refers to philosophical induction, suggesting a theory to explain observed facts, rather than mathematical induction, proving a property for all members of a well-ordered set.
Performing machine learning involves creating a model, which is trained on some training data and then can process additional data to make predictions. Various types of models have been used and researched for machine learning systems.
Artificial neural networks (ANNs), or connectionist systems, are computing systems vaguely inspired by the biological neural networks that constitute animal brains. Such systems "learn" to perform tasks by considering examples, generally without being programmed with any task-specific rules.
An ANN is a model based on a collection of connected units or nodes called "artificial neurons", which loosely model the neurons in a biological brain. Each connection, like the synapses in a biological brain, can transmit information, a "signal", from one artificial neuron to another. An artificial neuron that receives a signal can process it and then signal additional artificial neurons connected to it. In common ANN implementations, the signal at a connection between artificial neurons is a real number, and the output of each artificial neuron is computed by some non-linear function of the sum of its inputs. The connections between artificial neurons are called "edges". Artificial neurons and edges typically have a weight that adjusts as learning proceeds. The weight increases or decreases the strength of the signal at a connection. Artificial neurons may have a threshold such that the signal is only sent if the aggregate signal crosses that threshold. Typically, artificial neurons are aggregated into layers. Different layers may perform different kinds of transformations on their inputs. Signals travel from the first layer (the input layer) to the last layer (the output layer), possibly after traversing the layers multiple times.
The original goal of the ANN approach was to solve problems in the same way that a human brain would. However, over time, attention moved to performing specific tasks, leading to deviations from biology. Artificial neural networks have been used on a variety of tasks, including computer vision, speech recognition, machine translation, social network filtering, playing board and video games and medical diagnosis.
Deep learning consists of multiple hidden layers in an artificial neural network. This approach tries to model the way the human brain processes light and sound into vision and hearing. Some successful applications of deep learning are computer vision and speech recognition.[52]
Decision tree learning uses a decision tree as a predictive model to go from observations about an item (represented in the branches) to conclusions about the item's target value (represented in the leaves). It is one of the predictive modeling approaches used in statistics, data mining and machine learning. Tree models where the target variable can take a discrete set of values are called classification trees; in these tree structures, leaves represent class labels and branches represent conjunctions of features that lead to those class labels. Decision trees where the target variable can take continuous values (typically real numbers) are called regression trees. In decision analysis, a decision tree can be used to visually and explicitly represent decisions and decision making. In data mining, a decision tree describes data, but the resulting classification tree can be an input for decision making.
Support vector machines (SVMs), also known as support vector networks, are a set of related supervised learning methods used for classification and regression. Given a set of training examples, each marked as belonging to one of two categories, an SVM training algorithm builds a model that predicts whether a new example falls into one category or the other.[53]  An SVM training algorithm is a non-probabilistic, binary, linear classifier, although methods such as Platt scaling exist to use SVM in a probabilistic classification setting. In addition to performing linear classification, SVMs can efficiently perform a non-linear classification using what is called the kernel trick, implicitly mapping their inputs into high-dimensional feature spaces.
Regression analysis encompasses a large variety of statistical methods to estimate the relationship between input variables and their associated features. Its most common form is linear regression, where a single line is drawn to best fit the given data according to a mathematical criterion such as ordinary least squares. The latter is oftentimes extended by regularization (mathematics) methods to mitigate overfitting and high bias, as can be seen in ridge regression. When dealing with non-linear problems, go-to models include polynomial regression (e.g. used for trendline fitting in Microsoft Excel [54]), Logistic regression (often used in statistical classification) or even kernel regression, which introduces non-linearity by taking advantage of the kernel trick to implicitly map input variables to higher dimensional space. 
A Bayesian network, belief network or directed acyclic graphical model is a probabilistic graphical model that represents a set of random variables and their conditional independence with a directed acyclic graph (DAG). For example, a Bayesian network could represent the probabilistic relationships between diseases and symptoms. Given symptoms, the network can be used to compute the probabilities of the presence of various diseases. Efficient algorithms exist that perform inference and learning. Bayesian networks that model sequences of variables, like speech signals or protein sequences, are called dynamic Bayesian networks. Generalizations of Bayesian networks that can represent and solve decision problems under uncertainty are called influence diagrams.
A genetic algorithm (GA) is a search algorithm and heuristic technique that mimics the process of natural selection, using methods such as mutation and crossover to generate new genotypes in the hope of finding good solutions to a given problem. In machine learning, genetic algorithms were used in the 1980s and 1990s.[55][56] Conversely, machine learning techniques have been used to improve the performance of genetic and evolutionary algorithms.[57]
Usually, machine learning models require a lot of data in order for them to perform well. Usually, when training a machine learning model, one needs to collect a large, representative sample of data from a training set. Data from the training set can be as varied as a corpus of text, a collection of images, and data collected from individual users of a service. Overfitting is something to watch out for when training a machine learning model.
Federated learning is a new approach to training machine learning models that decentralizes the training process, allowing for users' privacy to be maintained by not needing to send their data to a centralized server. This also increases efficiency by decentralizing the training process to many devices. For example, Gboard uses federated machine learning to train search query prediction models on users' mobile phones without having to send individual searches back to Google.[58]
There are many applications for machine learning, including:
In 2006, the media-services provider Netflix held the first "Netflix Prize" competition to find a program to better predict user preferences and improve the accuracy on its existing Cinematch movie recommendation algorithm by at least 10%.  A joint team made up of researchers from AT&T Labs-Research in collaboration with the teams Big Chaos and Pragmatic Theory built an ensemble model to win the Grand Prize in 2009 for $1 million.[60] Shortly after the prize was awarded, Netflix realized that viewers' ratings were not the best indicators of their viewing patterns ("everything is a recommendation") and they changed their recommendation engine accordingly.[61] In 2010 The Wall Street Journal wrote about the firm Rebellion Research and their use of machine learning to predict the financial crisis.[62] In 2012, co-founder of Sun Microsystems, Vinod Khosla, predicted that 80% of medical doctors' jobs would be lost in the next two decades to automated machine learning medical diagnostic software.[63] In 2014, it was reported that a machine learning algorithm had been applied in the field of art history to study fine art paintings, and that it may have revealed previously unrecognized influences among artists.[64] In 2019 Springer Nature published the first research book created using machine learning.[65]
Although machine learning has been transformative in some fields, machine-learning programs often fail to deliver expected results.[66][67][68] Reasons for this are numerous: lack of (suitable) data, lack of access to the data, data bias, privacy problems, badly chosen tasks and algorithms, wrong tools and people, lack of resources, and evaluation problems.[69]
In 2018, a self-driving car from Uber failed to detect a pedestrian, who was killed after a collision.[70] Attempts to use machine learning in healthcare with the IBM Watson system failed to deliver even after years of time and billions of investment.[71][72]
Machine learning approaches in particular can suffer from different data biases. A machine learning system trained on current customers only may not be able to predict the needs of new customer groups that are not represented in the training data. When trained on man-made data, machine learning is likely to pick up the same constitutional and unconscious biases already present in society.[73] Language models learned from data have been shown to contain human-like biases.[74][75] Machine learning systems used for criminal risk assessment have been found to be biased against black people.[76][77] In 2015, Google photos would often tag black people as gorillas,[78] and in 2018 this still was not well resolved, but Google reportedly was still using the workaround to remove all gorillas from the training data, and thus was not able to recognize real gorillas at all.[79] Similar issues with recognizing non-white people have been found in many other systems.[80] In 2016, Microsoft tested a chatbot that learned from Twitter, and it quickly picked up racist and sexist language.[81] Because of such challenges, the effective use of machine learning may take longer to be adopted in other domains.[82] Concern for fairness in machine learning, that is, reducing bias in machine learning and propelling its use for human good is increasingly expressed by artificial intelligence scientists, including Fei-Fei Li, who reminds engineers that "There’s nothing artificial about AI...It’s inspired by people, it’s created by people, and—most importantly—it impacts people. It is a powerful tool we are only just beginning to understand, and that is a profound responsibility.”[83]
Classification machine learning models can be validated by accuracy estimation techniques like the Holdout method, which splits the data in a training and test set (conventionally 2/3 training set and 1/3 test set designation) and evaluates the performance of the training model on the test set. In comparison, the K-fold-cross-validation method randomly partitions the data into K subsets and then K experiments are performed each respectively considering 1 subset for evaluation and the remaining K-1 subsets for training the model. In addition to the holdout and cross-validation methods, bootstrap, which samples n instances with replacement from the dataset, can be used to assess model accuracy.[84]
In addition to overall accuracy, investigators frequently report sensitivity and specificity meaning True Positive Rate (TPR) and True Negative Rate (TNR) respectively. Similarly, investigators sometimes report the False Positive Rate (FPR) as well as the False Negative Rate (FNR). However, these rates are ratios that fail to reveal their numerators and denominators. The Total Operating Characteristic (TOC) is an effective method to express a model's diagnostic ability. TOC shows the numerators and denominators of the previously mentioned rates, thus TOC provides more information than the commonly used Receiver Operating Characteristic (ROC) and ROC's associated Area Under the Curve (AUC).[85]
Machine learning poses a host of ethical questions. Systems which are trained on datasets collected with biases may exhibit these biases upon use (algorithmic bias), thus digitizing cultural prejudices.[86] For example, using job hiring data from a firm with racist hiring policies may lead to a machine learning system duplicating the bias by scoring job applicants against similarity to previous successful applicants.[87][88] Responsible collection of data and documentation of algorithmic rules used by a system thus is a critical part of machine learning.
Because human languages contain biases, machines trained on language corpora will necessarily also learn these biases.[89][90]
Other forms of ethical challenges, not related to personal biases, are more seen in health care. There are concerns among health care professionals that these systems might not be designed in the public's interest but as income-generating machines. This is especially true in the United States where there is a long-standing ethical dilemma of improving health care, but also increasing profits. For example, the algorithms could be designed to provide patients with unnecessary tests or medication in which the algorithm's proprietary owners hold stakes. There is huge potential for machine learning in health care to provide professionals a great tool to diagnose, medicate, and even plan recovery paths for patients, but this will not happen until the personal biases mentioned previously, and these "greed" biases are addressed.[91]
Software suites containing a variety of machine learning algorithms include the following:
qXh$  The world is quietly being reshaped by machine learning. We no longer need to teach computers how to perform complex tasks like image recognition or text translation: instead, we build systems that let them learn how to do it themselves. “It’s not magic,” says Greg Corrado, a senior research scientist at Google. “It’s just a tool. But it’s a really important tool.” The most powerful form of machine learning being used today, called “deep learning”, builds a complex mathematical structure called a neural network based on vast quantities of data. Designed to be analogous to how a human brain works, neural networks themselves were first described in the 1930s. But it’s only in the last three or four years that computers have become powerful enough to use them effectively. Corrado says he thinks it is as big a change for tech as the internet was. “Before internet technologies, if you worked in computer science, networking was some weird thing that weirdos did. And now everyone, regardless of whether they’re an engineer or a software developer or a product designer or a CEO understands how internet connectivity shapes their product, shapes the market, what they could possibly build.” He says that same kind of transformation is going to happen with machine learning. “It ends up being something that everybody can do a little of. They don’t have to do the detailed things, but they need to understand ‘well, wait a minute, maybe we could do this if we had data to learn from.’” Google’s own implementation of the idea, an open-source software suite called TensorFlow, was built from the ground up to be useable by both the researchers at the company attempting to understand the powerful models they create, as well as the engineers who are already taking them, bottling them up, and using them to categorise photos or let people search with their voice. Machine learning is still a complex beast. Away from simplified playgrounds, there’s not much you can do with neural networks yourself unless you have a strong background in coding. But I wanted to put Conrado’s claims to the test: if machine learning will be something “everybody can do a little of” in the future, how close is it to that today? One of the nice things about the machine learning community right now is how open it is to sharing ideas and research. When Google made TensorFlow open to anyone to use, it wrote: “By sharing what we believe to be one of the best machine learning toolboxes in the world, we hope to create an open standard for exchanging research ideas and putting machine learning in products”. And it’s not alone in that: every major machine learning implementation is available for free to use and modify, meaning it’s possible to set up a simple machine intelligence with nothing more than a laptop and a web connection. Which is what I did. Following the lead of writer and technologist Robin Sloan, I trained a simple neural network on 119mb of Guardian leader columns. It wasn’t easy. Even with a detailed readme, it took me a few hours to set up a computer to the point where it could start learning from the corpus of text. And once it reached that point, I realised I had vastly underrated the amount of time it takes for a machine to learn. After running the training software for 30 minutes, and getting around 1% of the way through, I realised I would need a much faster computer. Finally got this running: snappy in-editor "autocomplete" powered by a neural net trained on old sci-fi stories. pic.twitter.com/Cu4GCZdUEl So I spent another few hours configuring a server on Amazon’s cloud to do the learning for me. It cost $.70 an hour, but meant that the whole thing was done in about 8 hours. I’m not the only one to play around with the technology. Quietly, starting a few years ago, Google itself has undergone a metamorphosis. The search giant has torn out the guts of some of its biggest services, from image search to voice recognition, and recreated them from the ground up. Now, it wants the rest of the world to follow suit. On 16 June, it announced that it was opening a dedicated Machine Learning group in its Zurich engineering office, the largest collection of Google developers outside of the US, to lead research into three areas: machine intelligence, natural language processing, and machine perception. That is, building systems that can think, listen, and see. But while computer scientists know enough about how to wrangle neural networks to use them to identify speech or create psychedelic images, they don’t really know all there is to know about how they actually work. They sort of just … do. Part of the job of Google DeepMind, the research arm which most famously led an algorithm to victory over a world champion in the ancient Asian board game Go, is to work out a little bit more about why and how they are so good. And the new machine learning group is straddling the line between research and product development, attempting to build new algorithms that can tackle unprecedented challenges. My own attempt to do the same didn’t go so well. The results were … not perfect. While Google’s machine learning demonstrations involve solving problems which were described as “virtually impossible” just two years ago, mine could barely string a sentence together. Following Sloan’s example, I set my model up to run as an autocomplete engine. I could write the first half-sentance of a theoretical Guardian editorial, and the system gets fed it as an input and asked what it thinks will come next. Don’t like that? Ask it for another response. I tried to use it to read Guardian editorials from a parallel universe. I used “Thursday’s momentous vote to stay in the EU was” as the seed, and tried to get the system to imagine what the rest of the sentence would look like: It’s terrible. Of course it’s terrible: if I could train a machine to write a convincing Guardian editorial, or even a convincing sentence extract from a Guardian editorial, in two days by copying a readme and fiddling around with complex software which I don’t really understand even after having successfully used it, then my job would be much less secure than it is. Hell, everyone’s jobs would be much less secure than they are.  I’m not even the first to fall at this hurdle: the Atlantic’s Adrienne LaFrance tried a similar experiment, also using Sloan’s kit, earlier in June, but was hampered by the size of her corpus. Half a million words, the total quantity of her writing from the Atlantic, isn’t quite enough for a machine to learn from, but the 20m sitting in the Guardian’s archive of editorials is better. (I could have run the system on every story in the archive, but it learns better if there’s a consistent tone and style for it to emulate – something leader columns, which are all written in the voice of the paper, have). While the results are unimpressive on the face of it, at the same time, however, they’re … kind of amazing. The specific package I used, called Torch-rnn, is designed for training character-level neural networks. That is, before it’s trained, it doesn’t even know the concept of a word, let alone have a specific vocabulary or understanding of English grammar. Now, I have a model that knows all those things. And it taught itself with nothing more than a huge quantity of Guardian editorials.  It still can’t actually create meaning. That makes sense: a Guardian editorial has meaning in relation to the real world, not as a collection of words existing in its own right. And so to properly train a neural network to write one, you’d also have to feed in information about the world, and then you’ve got less of a weekend project and more of a startup pitch. So it’s not surprising to see the number of startup pitches that do involve “deep learning” skyrocket. My inbox has consistently seen one or two a day for the past year, from an “online personal styling service” which uses deep learning to match people to clothes, to a “knowledge discovery engine” which aims to beat Google at its own game. Where the archetypal startup of 2008 was “x but on a phone” and the startup of 2014 was “uber but for x”, this year is the year of “doing x with machine learning”. And Google seems happy to be leading the way, not only with its own products, but also by making the tools which the rest of the ecosystem is relying on. But why now? Corrado has an answer. “The maths for deep learning was done in the 1980s and 1990s… but until now, computers were too slow for us to understand that the math worked well. “The fact that they’re getting faster and cheaper is part of what’s making this possible.” Right now, he says, doing machine learning yourself is like trying to go online by manually coding a TCP/IP stack.  But that’s going to change. It will get quicker, easier and more effective, and slowly move from something the engineers know about, to something the whole development team know about, then the whole tech industry, and then, eventually, everyone. And when it does, it’s going to change a lot else with it. • AlphaGo taught itself how to win, but without humans it would have run out of timeqX�  A powerful antibiotic that kills some of the most dangerous drug-resistant bacteria in the world has been discovered using artificial intelligence. The drug works in a different way to existing antibacterials and is the first of its kind to be found by setting AI loose on vast digital libraries of pharmaceutical compounds. Tests showed that the drug wiped out a range of antibiotic-resistant strains of bacteria, including Acinetobacter baumannii and Enterobacteriaceae, two of the three high-priority pathogens that the World Health Organization ranks as “critical” for new antibiotics to target. “In terms of antibiotic discovery, this is absolutely a first,” said Regina Barzilay, a senior researcher on the project and specialist in machine learning at Massachusetts Institute of Technology (MIT). “I think this is one of the more powerful antibiotics that has been discovered to date,” added James Collins, a bioengineer on the team at MIT. “It has remarkable activity against a broad range of antibiotic-resistant pathogens.” Antibiotic resistance arises when bacteria mutate and evolve to sidestep the mechanisms that antimicrobial drugs use to kill them. Without new antibiotics to tackle resistance, 10 million lives around the world could be at risk each year from infections by 2050, the Cameron government’s O’Neill report warned. To find new antibiotics, the researchers first trained a “deep learning” algorithm to identify the sorts of molecules that kill bacteria. To do this, they fed the program information on the atomic and molecular features of nearly 2,500 drugs and natural compounds, and how well or not the substance blocked the growth of the bug E coli. Once the algorithm had learned what molecular features made for good antibiotics, the scientists set it working on a library of more than 6,000 compounds under investigation for treating various human diseases. Rather than looking for any potential antimicrobials, the algorithm focused on compounds that looked effective but unlike existing antibiotics. This boosted the chances that the drugs would work in radical new ways that bugs had yet to develop resistance to. Jonathan Stokes, the first author of the study, said it took a matter of hours for the algorithm to assess the compounds and come up with some promising antibiotics. One, which the researchers named “halicin” after Hal, the astronaut-bothering AI in the film 2001: A Space Odyssey, looked particularly potent. Writing in the journal Cell, the researchers describe how they treated numerous drug-resistant infections with halicin, a compound that was originally developed to treat diabetes, but which fell by the wayside before it reached the clinic. Tests on bacteria collected from patients showed that halicin killed Mycobacterium tuberculosis, the bug that causes TB, and strains of Enterobacteriaceae that are resistant to carbapenems, a group of antibiotics that are considered the last resort for such infections. Halicin also cleared C difficile and multidrug-resistant Acinetobacter baumannii infections in mice. To hunt for more new drugs, the team next turned to a massive digital database of about 1.5bn compounds. They set the algorithm working on 107m of these. Three days later, the program returned a shortlist of 23 potential antibiotics, of which two appear to be particularly potent. The scientists now intend to search more of the database. Stokes said it would have been impossible to screen all 107m compounds by the conventional route of obtaining or making the substances and then testing them in the lab. “Being able to perform these experiments in the computer dramatically reduces the time and cost to look at these compounds,” he said. Barzilay now wants to use the algorithm to find antibiotics that are more selective in the bacteria they kill. This would mean that taking the antibiotic kills only the bugs causing an infection, and not all the healthy bacteria that live in the gut. More ambitiously, the scientists aim to use the algorithm to design potent new antibiotics from scratch. “The work really is remarkable,” said Jacob Durrant, who works on computer-aided drug design at the University of Pittsburgh. “Their approach highlights the power of computer-aided drug discovery. It would be impossible to physically test over 100m compounds for antibiotic activity.” “Given typical drug-development costs, in terms of both time and money, any method that can speed early-stage drug discovery has the potential to make a big impact,” he added.qXC  The following outline is provided as an overview of and topical guide to machine learning. Machine learning is a subfield of soft computing within computer science that evolved from the study of pattern recognition and computational learning theory in artificial intelligence.[1] In 1959, Arthur Samuel defined machine learning as a "field of study that gives computers the ability to learn without being explicitly programmed".[2] Machine learning explores the study and construction of algorithms that can learn from and make predictions on data.[3] Such algorithms operate by building a model from an example training set of input observations in order to make data-driven predictions or decisions expressed as outputs, rather than following strictly static program instructions.
Subfields of machine learning
Cross-disciplinary fields involving machine learning
Applications of machine learning
Machine learning hardware
Machine learning tools   (list)
Machine learning framework
Proprietary machine learning frameworks
Open source machine learning frameworks
Machine learning library   
Machine learning algorithm
Machine learning method   (list)
Dimensionality reduction
Ensemble learning
Meta learning
Reinforcement learning
Supervised learning
Bayesian statistics
Decision tree algorithm
Linear classifier
Unsupervised learning
Artificial neural network
Association rule learning
Hierarchical clustering
Cluster analysis
Anomaly detection
Semi-supervised learning
Deep learning
History of machine learning
Machine learning projects
Machine learning organizations
Books about machine learning
qX��  Deep learning  (also known as deep structured learning or differential programming) is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised.[1][2][3]
Deep learning architectures such as deep neural networks, deep belief networks, recurrent neural networks and convolutional neural networks have been applied to fields including computer vision, speech recognition, natural language processing, audio recognition, social network filtering, machine translation, bioinformatics, drug design, medical image analysis, material inspection and board game programs, where they have produced results comparable to and in some cases surpassing human expert performance.[4][5][6]
Artificial neural networks (ANNs) were inspired by information processing and distributed communication nodes in biological systems. ANNs have various differences from biological brains.  Specifically, neural networks tend to be static and symbolic, while the biological brain of most living organisms is dynamic (plastic) and analog.[7][8][9]
Deep learning is a class of machine learning algorithms that[11](pp199–200) uses multiple layers to progressively extract higher level features from the raw input. For example, in image processing, lower layers may identify edges, while higher layers may identify the concepts relevant to a human such as digits or letters or faces.
Most modern deep learning models are based on artificial neural networks, specifically, Convolutional Neural Networks (CNN)s, although they can also include propositional formulas or latent variables organized layer-wise in deep generative models such as the nodes in deep belief networks and deep Boltzmann machines.[12]
In deep learning, each level learns to transform its input data into a slightly more abstract and composite representation. In an image recognition application, the raw input may be a matrix of pixels; the first representational layer may abstract the pixels and encode edges; the second layer may compose and encode arrangements of edges; the third layer may encode a nose and eyes; and the fourth layer may recognize that the image contains a face. Importantly, a deep learning process can learn which features to optimally place in which level on its own. (Of course, this does not completely eliminate the need for hand-tuning; for example, varying numbers of layers and layer sizes can provide different degrees of abstraction.)[1][13]
The word "deep" in "deep learning" refers to the number of layers through which the data is transformed. More precisely, deep learning systems have a substantial credit assignment path (CAP) depth. The CAP is the chain of transformations from input to output. CAPs describe potentially causal connections between input and output. For a feedforward neural network, the depth of the CAPs is that of the network and is the number of hidden layers plus one (as the output layer is also parameterized). For recurrent neural networks, in which a signal may propagate through a layer more than once, the CAP depth is potentially unlimited.[2] No universally agreed upon threshold of depth divides shallow learning from deep learning, but most researchers agree that deep learning involves CAP depth higher than 2. CAP of depth 2 has been shown to be a universal approximator in the sense that it can emulate any function.[14] Beyond that, more layers do not add to the function approximator ability of the network. Deep models (CAP > 2) are able to extract better features than shallow models and hence, extra layers help in learning the features effectively.
Deep learning architectures can be constructed with a greedy layer-by-layer method.[15] Deep learning helps to disentangle these abstractions and pick out which features improve performance.[1]
For supervised learning tasks, deep learning methods eliminate feature engineering, by translating the data into compact intermediate representations akin to principal components, and derive layered structures that remove redundancy in representation.
Deep learning algorithms can be applied to unsupervised learning tasks. This is an important benefit because unlabeled data are more abundant than the labeled data. Examples of deep structures that can be trained in an unsupervised manner are neural history compressors[16] and deep belief networks.[1][17]
Deep neural networks are generally interpreted in terms of the universal approximation theorem[18][19][20][21][22][23] or probabilistic inference.[11][12][1][2][17][24][25]
The classic universal approximation theorem concerns the capacity of feedforward neural networks with a single hidden layer of finite size to approximate continuous functions.[18][19][20][21][22] In 1989, the first proof was published by George Cybenko for sigmoid activation functions[19] and was generalised to feed-forward multi-layer architectures in 1991 by Kurt Hornik.[20] Recent work also showed that universal approximation also holds for non-bounded activation functions such as the rectified linear unit.[26]
The universal approximation theorem for deep neural networks concerns the capacity of networks with bounded width but the depth is allowed to grow. Lu et al.[23] proved that if the width of a deep neural network with ReLU activation is strictly larger than the input dimension, then the network can approximate any Lebesgue integrable function; If the width is smaller or equal to the input dimension, then deep neural network is not a universal approximator.
The probabilistic interpretation[24] derives from the field of machine learning. It features inference,[11][12][1][2][17][24] as well as the optimization concepts of training and testing, related to fitting and generalization, respectively. More specifically, the probabilistic interpretation considers the activation nonlinearity as a cumulative distribution function.[24] The probabilistic interpretation led to the introduction of dropout as regularizer in neural networks.[27] The probabilistic interpretation was introduced by researchers including Hopfield, Widrow and Narendra and popularized in surveys such as the one by Bishop.[28]
The term Deep Learning was introduced to the machine learning community by Rina Dechter in 1986,[29][16] and to artificial neural networks by Igor Aizenberg and colleagues in 2000, in the context of Boolean threshold neurons.[30][31]
The first general, working learning algorithm for supervised, deep, feedforward, multilayer perceptrons was published by Alexey Ivakhnenko and Lapa in 1967.[32] A 1971 paper described already a deep network with 8 layers trained by the group method of data handling algorithm.[33]
Other deep learning working architectures, specifically those built for computer vision, began with the Neocognitron introduced by Kunihiko Fukushima in 1980.[34] In 1989, Yann LeCun et al. applied the standard backpropagation algorithm, which had been around as the reverse mode of automatic differentiation since 1970,[35][36][37][38] to a deep neural network with the purpose of recognizing handwritten ZIP codes on mail. While the algorithm worked, training required 3 days.[39]
By 1991 such systems were used for recognizing isolated 2-D hand-written digits, while recognizing 3-D objects was done by matching 2-D images with a handcrafted 3-D object model. Weng et al. suggested that a human brain does not use a monolithic 3-D object model and in 1992 they published Cresceptron,[40][41][42] a method for performing 3-D object recognition in cluttered scenes. Because it directly used natural images, Cresceptron started the beginning of general-purpose visual learning for natural 3D worlds. Cresceptron is a cascade of layers similar to Neocognitron. But while Neocognitron required a human programmer to hand-merge features, Cresceptron learned an open number of features in each layer without supervision, where each feature is represented by a convolution kernel. Cresceptron segmented each learned object from a cluttered scene through back-analysis through the network. Max pooling, now often adopted by deep neural networks (e.g. ImageNet tests), was first used in Cresceptron to reduce the position resolution by a factor of (2x2) to 1 through the cascade for better generalization.
In 1994, André de Carvalho, together with Mike Fairhurst and David Bisset, published experimental results of a multi-layer boolean neural network, also known as a weightless neural network, composed of a 3-layers self-organising feature extraction neural network module (SOFT) followed by a multi-layer classification neural network module (GSN), which were independently trained. Each layer in the feature extraction module extracted features with growing complexity regarding the previous layer.[43]
In 1995, Brendan Frey demonstrated that it was possible to train (over two days) a network containing six fully connected layers and several hundred hidden units using the wake-sleep algorithm, co-developed with Peter Dayan and Hinton.[44] Many factors contribute to the slow speed, including the vanishing gradient problem analyzed in 1991 by Sepp Hochreiter.[45][46]
Simpler models that use task-specific handcrafted features such as Gabor filters and support vector machines (SVMs) were a popular choice in the 1990s and 2000s, because of artificial neural network's (ANN) computational cost and a lack of understanding of how the brain wires its biological networks.
Both shallow and deep learning (e.g., recurrent nets) of ANNs have been explored for many years.[47][48][49] These methods never outperformed non-uniform internal-handcrafting Gaussian mixture model/Hidden Markov model (GMM-HMM) technology based on generative models of speech trained discriminatively.[50] Key difficulties have been analyzed, including gradient diminishing[45] and weak temporal correlation structure in neural predictive models.[51][52] Additional difficulties were the lack of training data and limited computing power.
Most speech recognition researchers moved away from neural nets to pursue generative modeling. An exception was at SRI International in the late 1990s. Funded by the US government's NSA and DARPA, SRI studied deep neural networks in speech and speaker recognition. The speaker recognition team led by Larry Heck reported significant success with deep neural networks in speech processing in the 1998 National Institute of Standards and Technology Speaker Recognition evaluation.[53] The SRI deep neural network was then deployed in the Nuance Verifier, representing the first major industrial application of deep learning.[54]
The principle of elevating "raw" features over hand-crafted optimization was first explored successfully in the architecture of deep autoencoder on the "raw" spectrogram or linear filter-bank features in the late 1990s,[54] showing its superiority over the Mel-Cepstral features that contain stages of fixed transformation from spectrograms. The raw features of speech, waveforms, later produced excellent larger-scale results.[55]
Many aspects of speech recognition were taken over by a deep learning method called long short-term memory (LSTM), a recurrent neural network published by Hochreiter and Schmidhuber in 1997.[56] LSTM RNNs avoid the vanishing gradient problem and can learn "Very Deep Learning" tasks[2] that require memories of events that happened thousands of discrete time steps before, which is important for speech. In 2003, LSTM started to become competitive with traditional speech recognizers on certain tasks.[57] Later it was combined with connectionist temporal classification (CTC)[58] in stacks of LSTM RNNs.[59] In 2015, Google's speech recognition reportedly experienced a dramatic performance jump of 49% through CTC-trained LSTM, which they made available through Google Voice Search.[60]
In 2006, publications by Geoff Hinton, Ruslan Salakhutdinov, Osindero and Teh[61]
[62][63] showed how a many-layered feedforward neural network could be effectively pre-trained one layer at a time, treating each layer in turn as an unsupervised restricted Boltzmann machine, then fine-tuning it using supervised backpropagation.[64] The papers referred to learning for deep belief nets.
Deep learning is part of state-of-the-art systems in various disciplines, particularly computer vision and automatic speech recognition (ASR). Results on commonly used evaluation sets such as TIMIT (ASR) and MNIST (image classification), as well as a range of large-vocabulary speech recognition tasks have steadily improved.[65][66][67] Convolutional neural networks (CNNs) were superseded for ASR by CTC[58] for LSTM.[56][60][68][69][70][71][72] but are more successful in computer vision.
The impact of deep learning in industry began in the early 2000s, when CNNs already processed an estimated 10% to 20% of all the checks written in the US, according to Yann LeCun.[73] Industrial applications of deep learning to large-scale speech recognition started around 2010.
The 2009 NIPS Workshop on Deep Learning for Speech Recognition[74] was motivated by the limitations of deep generative models of speech, and the possibility that given more capable hardware and large-scale data sets that deep neural nets (DNN) might become practical. It was believed that pre-training DNNs using generative models of deep belief nets (DBN) would overcome the main difficulties of neural nets.[75] However, it was discovered that replacing pre-training with large amounts of training data for straightforward backpropagation when using DNNs with large, context-dependent output layers produced error rates dramatically lower than then-state-of-the-art Gaussian mixture model (GMM)/Hidden Markov Model (HMM) and also than more-advanced generative model-based systems.[65][76] The nature of the recognition errors produced by the two types of systems was characteristically different,[77][74] offering technical insights into how to integrate deep learning into the existing highly efficient, run-time speech decoding system deployed by all major speech recognition systems.[11][78][79] Analysis around 2009-2010, contrasted the GMM (and other generative speech models) vs. DNN models, stimulated early industrial investment in deep learning for speech recognition,[77][74] eventually leading to pervasive and dominant use in that industry. That analysis was done with comparable performance (less than 1.5% in error rate) between discriminative DNNs and generative models.[65][77][75][80]
In 2010, researchers extended deep learning from TIMIT to large vocabulary speech recognition, by adopting large output layers of the DNN based on context-dependent HMM states constructed by decision trees.[81][82][83][78]
Advances in hardware have enabled renewed interest in deep learning. In 2009, Nvidia was involved in what was called the “big bang” of deep learning, “as deep-learning neural networks were trained with Nvidia graphics processing units (GPUs).”[84] That year, Google Brain used Nvidia GPUs to create capable DNNs. While there, Andrew Ng determined that GPUs could increase the speed of deep-learning systems by about 100 times.[85] In particular, GPUs are well-suited for the matrix/vector computations involved in machine learning.[86][87][88] GPUs speed up training algorithms by orders of magnitude, reducing running times from weeks to days.[89][90] Further, specialized hardware and algorithm optimizations can be used for efficient processing of deep learning models.[91]
In 2012, a team led by George E. Dahl won the "Merck Molecular Activity Challenge" using multi-task deep neural networks to predict the biomolecular target of one drug.[92][93] In 2014, Hochreiter's group used deep learning to detect off-target and toxic effects of environmental chemicals in nutrients, household products and drugs and won the "Tox21 Data Challenge" of NIH, FDA and NCATS.[94][95][96]
Significant additional impacts in image or object recognition were felt from 2011 to 2012. Although CNNs trained by backpropagation had been around for decades, and GPU implementations of NNs for years, including CNNs, fast implementations of CNNs with max-pooling on GPUs in the style of Ciresan and colleagues were needed to progress on computer vision.[86][88][39][97][2] In 2011, this approach achieved for the first time superhuman performance in a visual pattern recognition contest. Also in 2011, it won the ICDAR Chinese handwriting contest, and in May 2012, it won the ISBI image segmentation contest.[98] Until 2011, CNNs did not play a major role at computer vision conferences, but in June 2012, a paper by Ciresan et al. at the leading conference CVPR[4] showed how max-pooling CNNs on GPU can dramatically improve many vision benchmark records. In October 2012, a similar system by Krizhevsky et al.[5] won the large-scale ImageNet competition by a significant margin over shallow machine learning methods. In November 2012, Ciresan et al.'s system also won the ICPR contest on analysis of large medical images for cancer detection, and in the following year also the MICCAI Grand Challenge on the same topic.[99] In 2013 and 2014, the error rate on the ImageNet task using deep learning was further reduced, following a similar trend in large-scale speech recognition. The Wolfram Image Identification project publicized these improvements.[100]
Image classification was then extended to the more challenging task of generating descriptions (captions) for images, often as a combination of CNNs and LSTMs.[101][102][103][104]
Some researchers assess that the October 2012 ImageNet victory anchored the start of a "deep learning revolution" that has transformed the AI industry.[105]
In March 2019, Yoshua Bengio, Geoffrey Hinton and Yann LeCun were awarded the Turing Award for conceptual and engineering breakthroughs that have made deep neural networks a critical component of computing.
Artificial neural networks (ANNs) or connectionist systems are computing systems inspired by the biological neural networks that constitute animal brains. Such systems learn (progressively improve their ability) to do tasks by considering examples, generally without task-specific programming. For example, in image recognition, they might learn to identify images that contain cats by analyzing example images that have been manually labeled as "cat" or "no cat" and using the analytic results to identify cats in other images. They have found most use in applications difficult to express with a traditional computer algorithm using rule-based programming.
An ANN is based on a collection of connected units called artificial neurons, (analogous to biological neurons in a biological brain). Each connection (synapse) between neurons can transmit a signal to another neuron. The receiving (postsynaptic) neuron can process the signal(s) and then signal downstream neurons connected to it. Neurons may have state, generally represented by real numbers, typically between 0 and 1. Neurons and synapses may also have a weight that varies as learning proceeds, which can increase or decrease the strength of the signal that it sends downstream.
Typically, neurons are organized in layers. Different layers may perform different kinds of transformations on their inputs. Signals travel from the first (input), to the last (output) layer, possibly after traversing the layers multiple times.
The original goal of the neural network approach was to solve problems in the same way that a human brain would. Over time, attention focused on matching specific mental abilities, leading to deviations from biology such as backpropagation, or passing information in the reverse direction and adjusting the network to reflect that information.
Neural networks have been used on a variety of tasks, including computer vision, speech recognition, machine translation, social network filtering, playing board and video games and medical diagnosis.
As of 2017, neural networks typically have a few thousand to a few million units and millions of connections. Despite this number being several order of magnitude less than the number of neurons on a human brain, these networks can perform many tasks at a level beyond that of humans (e.g., recognizing faces, playing "Go"[106] ).
A deep neural network (DNN) is an artificial neural network (ANN) with multiple layers between the input and output layers.[12][2] The DNN finds the correct mathematical manipulation to turn the input into the output, whether it be a linear relationship or a non-linear relationship. The network moves through the layers calculating the probability of each output. For example, a DNN that is trained to recognize dog breeds will go over the given image and calculate the probability that the dog in the image is a certain breed. The user can review the results and select which probabilities the network should display (above a certain threshold, etc.) and return the proposed label. Each mathematical manipulation as such is considered a layer, and complex DNN have many layers, hence the name "deep" networks.
DNNs can model complex non-linear relationships. DNN architectures generate compositional models where the object is expressed as a layered composition of primitives.[107] The extra layers enable composition of features from lower layers, potentially modeling complex data with fewer units than a similarly performing shallow network.[12]
Deep architectures include many variants of a few basic approaches. Each architecture has found success in specific domains. It is not always possible to compare the performance of multiple architectures, unless they have been evaluated on the same data sets.
DNNs are typically feedforward networks in which data flows from the input layer to the output layer without looping back. At first, the DNN creates a map of virtual neurons and assigns random numerical values, or "weights", to connections between them. The weights and inputs are multiplied and return an output between 0 and 1. If the network did not accurately recognize a particular pattern, an algorithm would adjust the weights.[108] That way the algorithm can make certain parameters more influential, until it determines the correct mathematical manipulation to fully process the data.
Recurrent neural networks (RNNs), in which data can flow in any direction, are used for applications such as language modeling.[109][110][111][112][113] Long short-term memory is particularly effective for this use.[56][114]
Convolutional deep neural networks (CNNs) are used in computer vision.[115] CNNs also have been applied to acoustic modeling for automatic speech recognition (ASR).[72]
As with ANNs, many issues can arise with naively trained DNNs. Two common issues are overfitting and computation time.
DNNs are prone to overfitting because of the added layers of abstraction, which allow them to model rare dependencies in the training data. Regularization methods such as Ivakhnenko's unit pruning[33] or weight decay (




ℓ

2




{\displaystyle \ell _{2}}

-regularization) or sparsity (




ℓ

1




{\displaystyle \ell _{1}}

-regularization) can be applied during training to combat overfitting.[116] Alternatively dropout regularization randomly omits units from the hidden layers during training. This helps to exclude rare dependencies.[117] Finally, data can be augmented via methods such as cropping and rotating such that smaller training sets can be increased in size to reduce the chances of overfitting.[118]
DNNs must consider many training parameters, such as the size (number of layers and number of units per layer), the learning rate, and initial weights. Sweeping through the parameter space for optimal parameters may not be feasible due to the cost in time and computational resources. Various tricks, such as batching (computing the gradient on several training examples at once rather than individual examples)[119] speed up computation. Large processing capabilities of many-core architectures (such as GPUs or the Intel Xeon Phi) have produced significant speedups in training, because of the suitability of such processing architectures for the matrix and vector computations.[120][121]
Alternatively, engineers may look for other types of neural networks with more straightforward and convergent training algorithms. CMAC (cerebellar model articulation controller) is one such kind of neural network. It doesn't require learning rates or randomized initial weights for CMAC. The training process can be guaranteed to converge in one step with a new batch of data, and the computational complexity of the training algorithm is linear with respect to the number of neurons involved.[122][123]
Large-scale automatic speech recognition is the first and most convincing successful case of deep learning. LSTM RNNs can learn "Very Deep Learning" tasks[2] that involve multi-second intervals containing speech events separated by thousands of discrete time steps, where one time step corresponds to about 10 ms. LSTM with forget gates[114] is competitive with traditional speech recognizers on certain tasks.[57]
The initial success in speech recognition was based on small-scale recognition tasks based on TIMIT. The data set contains 630 speakers from eight major dialects of American English, where each speaker reads 10 sentences.[124] Its small size lets many configurations be tried. More importantly, the TIMIT task concerns phone-sequence recognition, which, unlike word-sequence recognition, allows weak phone bigram language models. This lets the strength of the acoustic modeling aspects of speech recognition be more easily analyzed. The error rates listed below, including these early results and measured as percent phone error rates (PER), have been summarized since 1991.
The debut of DNNs for speaker recognition in the late 1990s and speech recognition around 2009-2011 and of LSTM around 2003-2007, accelerated progress in eight major areas:[11][80][78]
All major commercial speech recognition systems (e.g., Microsoft Cortana, Xbox, Skype Translator, Amazon Alexa, Google Now, Apple Siri, Baidu and iFlyTek voice search, and a range of Nuance speech products, etc.) are based on deep learning.[11][129][130][131]
A common evaluation set for image classification is the MNIST database data set. MNIST is composed of handwritten digits and includes 60,000 training examples and 10,000 test examples. As with TIMIT, its small size lets users test multiple configurations. A comprehensive list of results on this set is available.[132]
Deep learning-based image recognition has become "superhuman", producing more accurate results than human contestants. This first occurred in 2011.[133]
Deep learning-trained vehicles now interpret 360° camera views.[134] Another example is Facial Dysmorphology Novel Analysis (FDNA) used to analyze cases of human malformation connected to a large database of genetic syndromes.
Closely related to the progress that has been made in image recognition is the increasing application of deep learning techniques to various visual art tasks. DNNs have proven themselves capable, for example, of a) identifying the style period of a given painting, b) Neural Style Transfer - capturing the style of a given artwork and applying it in a visually pleasing manner to an arbitrary photograph or video, and c) generating striking imagery based on random visual input fields.[135][136]
Neural networks have been used for implementing language models since the early 2000s.[109][137] LSTM helped to improve machine translation and language modeling.[110][111][112]
Other key techniques in this field are negative sampling[138] and word embedding. Word embedding, such as word2vec, can be thought of as a representational layer in a deep learning architecture that transforms an atomic word into a positional representation of the word relative to other words in the dataset; the position is represented as a point in a vector space. Using word embedding as an RNN input layer allows the network to parse sentences and phrases using an effective compositional vector grammar. A compositional vector grammar can be thought of as probabilistic context free grammar (PCFG) implemented by an RNN.[139] Recursive auto-encoders built atop word embeddings can assess sentence similarity and detect paraphrasing.[139] Deep neural architectures provide the best results for constituency parsing,[140] sentiment analysis,[141] information retrieval,[142][143] spoken language understanding,[144] machine translation,[110][145] contextual entity linking,[145] writing style recognition,[146] Text classification and others.[147]
Recent developments generalize word embedding to sentence embedding.
Google Translate (GT) uses a large end-to-end long short-term memory network.[148][149][150][151][152][153] Google Neural Machine Translation (GNMT) uses an example-based machine translation method in which the system "learns from millions of examples."[149] It translates "whole sentences at a time, rather than pieces. Google Translate supports over one hundred languages.[149] The network encodes the "semantics of the sentence rather than simply memorizing phrase-to-phrase translations".[149][154] GT uses English as an intermediate between most language pairs.[154]
A large percentage of candidate drugs fail to win regulatory approval. These failures are caused by insufficient efficacy (on-target effect), undesired interactions (off-target effects), or unanticipated toxic effects.[155][156] Research has explored use of deep learning to predict the biomolecular targets,[92][93] off-targets, and toxic effects of environmental chemicals in nutrients, household products and drugs.[94][95][96]
AtomNet is a deep learning system for structure-based rational drug design.[157] AtomNet was used to predict novel candidate biomolecules for disease targets such as the Ebola virus[158] and multiple sclerosis.[159][160]
In 2019 generative neural networks were used to produce molecules that were validated experimentally all the way into mice.[161][162]
Deep reinforcement learning has been used to approximate the value of possible direct marketing actions, defined in terms of RFM variables. The estimated value function was shown to have a natural interpretation as customer lifetime value.[163]
Recommendation systems have used deep learning to extract meaningful features for a latent factor model for content-based music and journal recommendations.[164][165] Multiview deep learning has been applied for learning user preferences from multiple domains.[166] The model uses a hybrid collaborative and content-based approach and enhances recommendations in multiple tasks.
An autoencoder ANN was used in bioinformatics, to predict gene ontology annotations and gene-function relationships.[167]
In medical informatics, deep learning was used to predict sleep quality based on data from wearables[168] and predictions of health complications from electronic health record data.[169] Deep learning has also showed efficacy in healthcare.[170]
Deep learning has been shown to produce competitive results in medical application such as cancer cell classification, lesion detection, organ segmentation and image enhancement[171][172]
Finding the appropriate mobile audience for mobile advertising is always challenging, since many data points must be considered and analyzed before a target segment can be created and used in ad serving by any ad server.[173] Deep learning has been used to interpret large, many-dimensioned advertising datasets. Many data points are collected during the request/serve/click internet advertising cycle. This information can form the basis of machine learning to improve ad selection.
Deep learning has been successfully applied to inverse problems such as denoising, super-resolution, inpainting, and film colorization.[174] These applications include learning methods such as "Shrinkage Fields for Effective Image Restoration"[175] which trains on an image dataset, and Deep Image Prior, which trains on the image that needs restoration.
Deep learning is being successfully applied to financial fraud detection and anti-money laundering. "Deep anti-money laundering detection system can spot and recognize relationships and similarities between data and, further down the road, learn to detect anomalies or classify and predict specific events". The solution leverages both supervised learning techniques, such as the classification of suspicious transactions, and unsupervised learning, e.g. anomaly detection.
[176]
The United States Department of Defense applied deep learning to train robots in new tasks through observation.[177]
Deep learning is closely related to a class of theories of brain development (specifically, neocortical development) proposed by cognitive neuroscientists in the early 1990s.[178][179][180][181] These developmental theories were instantiated in computational models, making them predecessors of deep learning systems. These developmental models share the property that various proposed learning dynamics in the brain (e.g., a wave of nerve growth factor) support the self-organization somewhat analogous to the neural networks utilized in deep learning models. Like the neocortex, neural networks employ a hierarchy of layered filters in which each layer considers information from a prior layer (or the operating environment), and then passes its output (and possibly the original input), to other layers. This process yields a self-organizing stack of transducers, well-tuned to their operating environment. A 1995 description stated, "...the infant's brain seems to organize itself under the influence of waves of so-called trophic-factors ... different regions of the brain become connected sequentially, with one layer of tissue maturing before another and so on until the whole brain is mature."[182]
A variety of approaches have been used to investigate the plausibility of deep learning models from a neurobiological perspective. On the one hand, several variants of the backpropagation algorithm have been proposed in order to increase its processing realism.[183][184] Other researchers have argued that unsupervised forms of deep learning, such as those based on hierarchical generative models and deep belief networks, may be closer to biological reality.[185][186] In this respect, generative neural network models have been related to neurobiological evidence about sampling-based processing in the cerebral cortex.[187]
Although a systematic comparison between the human brain organization and the neuronal encoding in deep networks has not yet been established, several analogies have been reported. For example, the computations performed by deep learning units could be similar to those of actual neurons[188][189] and neural populations.[190] Similarly, the representations developed by deep learning models are similar to those measured in the primate visual system[191] both at the single-unit[192] and at the population[193] levels.
Facebook's AI lab performs tasks such as automatically tagging uploaded pictures with the names of the people in them.[194]
Google's DeepMind Technologies developed a system capable of learning how to play Atari video games using only pixels as data input. In 2015 they demonstrated their AlphaGo system, which learned the game of Go well enough to beat a professional Go player.[195][196][197] Google Translate uses a neural network to translate between more than 100 languages.
In 2015, Blippar demonstrated a mobile augmented reality application that uses deep learning to recognize objects in real time.[198]
In 2017, Covariant.ai was launched, which focuses on integrating deep learning into factories.[199]
As of 2008,[200] researchers at The University of Texas at Austin (UT) developed a machine learning framework called Training an Agent Manually via Evaluative Reinforcement, or TAMER, which proposed new methods for robots or computer programs to learn how to perform tasks by interacting with a human instructor.[177] First developed as TAMER, a new algorithm called Deep TAMER was later introduced in 2018 during a collaboration between U.S. Army Research Laboratory (ARL) and UT researchers. Deep TAMER used deep learning to provide a robot the ability to learn new tasks through observation.[177] Using Deep TAMER, a robot learned a task with a human trainer, watching video streams or observing a human perform a task in-person. The robot later practiced the task with the help of some coaching from the trainer, who provided feedback such as “good job” and “bad job.”[201]
Deep learning has attracted both criticism and comment, in some cases from outside the field of computer science.
A main criticism concerns the lack of theory surrounding some methods.[202] Learning in the most common deep architectures is implemented using well-understood gradient descent. However, the theory surrounding other algorithms, such as contrastive divergence is less clear.[citation needed] (e.g., Does it converge? If so, how fast? What is it approximating?) Deep learning methods are often looked at as a black box, with most confirmations done empirically, rather than theoretically.[203]

Others point out that deep learning should be looked at as a step towards realizing strong AI, not as an all-encompassing solution. Despite the power of deep learning methods, they still lack much of the functionality needed for realizing this goal entirely. Research psychologist Gary Marcus noted:"Realistically, deep learning is only part of the larger challenge of building intelligent machines. Such techniques lack ways of representing causal relationships (...) have no obvious ways of performing logical inferences, and they are also still a long way from integrating abstract knowledge, such as information about what objects are, what they are for, and how they are typically used. The most powerful A.I. systems, like Watson (...) use techniques like deep learning as just one element in a very complicated ensemble of techniques, ranging from the statistical technique of Bayesian inference to deductive reasoning."[204]As an alternative to this emphasis on the limits of deep learning, one author speculated that it might be possible to train a machine vision stack to perform the sophisticated task of discriminating between "old master" and amateur figure drawings, and hypothesized that such a sensitivity might represent the rudiments of a non-trivial machine empathy.[205] This same author proposed that this would be in line with anthropology, which identifies a concern with aesthetics as a key element of behavioral modernity.[206]
In further reference to the idea that artistic sensitivity might inhere within relatively low levels of the cognitive hierarchy, a published series of graphic representations of the internal states of deep (20-30 layers) neural networks attempting to discern within essentially random data the images on which they were trained[207] demonstrate a visual appeal: the original research notice received well over 1,000 comments, and was the subject of what was for a time the most frequently accessed article on The Guardian's[208] website.
Some deep learning architectures display problematic behaviors,[209] such as confidently classifying unrecognizable images as belonging to a familiar category of ordinary images[210] and misclassifying minuscule perturbations of correctly classified images.[211] Goertzel hypothesized that these behaviors are due to limitations in their internal representations and that these limitations would inhibit integration into heterogeneous multi-component artificial general intelligence (AGI) architectures.[209] These issues may possibly be addressed by deep learning architectures that internally form states homologous to image-grammar[212] decompositions of observed entities and events.[209] Learning a grammar (visual or linguistic) from training data would be equivalent to restricting the system to commonsense reasoning that operates on concepts in terms of grammatical production rules and is a basic goal of both human language acquisition[213] and artificial intelligence (AI).[214]
As deep learning moves from the lab into the world, research and experience shows that artificial neural networks are vulnerable to hacks and deception.[215] By identifying patterns that these systems use to function, attackers can modify inputs to ANNs in such a way that the ANN finds a match that human observers would not recognize. For example, an attacker can make subtle changes to an image such that the ANN finds a match even though the image looks to a human nothing like the search target. Such a manipulation is termed an “adversarial attack.”[216] In 2016 researchers used one ANN to doctor images in trial and error fashion, identify another's focal points and thereby generate images that deceived it. The modified images looked no different to human eyes. Another group showed that printouts of doctored images then photographed successfully tricked an image classification system.[217] One defense is reverse image search, in which a possible fake image is submitted to a site such as TinEye that can then find other instances of it. A refinement is to search using only parts of the image, to identify images from which that piece may have been taken.[218]
Another group showed that certain psychedelic spectacles could fool a facial recognition system into thinking ordinary people were celebrities, potentially allowing one person to impersonate another. In 2017 researchers added stickers to stop signs and caused an ANN to misclassify them.[217]
ANNs can however be further trained to detect attempts at deception, potentially leading attackers and defenders into an arms race similar to the kind that already defines the malware defense industry. ANNs have been trained to defeat ANN-based anti-malware software by repeatedly attacking a defense with malware that was continually altered by a genetic algorithm until it tricked the anti-malware while retaining its ability to damage the target.[217]
Another group demonstrated that certain sounds could make the Google Now voice command system open a particular web address that would download malware.[217]
In “data poisoning,” false data is continually smuggled into a machine learning system's training set to prevent it from achieving mastery.[217]
Most Deep Learning systems rely on training and verification data that is generated and/or annotated by humans. It has been argued in media philosophy that not only low-payed clickwork (e.g. on Amazon Mechanical Turk) is regularly deployed for this purpose, but also implicit forms of human microwork that are often not recognized as such.[219] The philosopher Rainer Mühlhoff distinguishes five types of "machinic capture" of human microwork to generate training data: (1) gamification (the embedding of annotation or computation tasks in the flow of a game), (2) "trapping and tracking" (e.g. CAPTCHAs for image recognition or click-tracking on Google search results pages), (3) exploitation of social motivations (e.g. tagging faces on Facebook to obtain labeled facial images), (4) information mining (e.g. by leveraging quantified-self devices such as activity trackers) and (5) clickwork.[219] Mühlhoff argues that in most commercial end-user applications of Deep Learning such as Facebook's face recognition system, the need for training data does not stop once an ANN is trained. Rather, there is a continued demand for human-generated verification data to constantly calibrate and update the ANN. For this purpose Facebook introduced the feature that once a user is automatically recognized in an image, they receive a notification. They can choose whether of not they like to be publicly labeled on the image, or tell Facebook that it is not them in the picture.[220] This user interface is a mechanism to generate "a constant stream of  verification data"[219] to further train the network in real-time. As Mühlhoff argues, involvement of human users to generate training and verification data is so typical for most commercial end-user applications of Deep Learning that such systems may be referred to as "human-aided artificial intelligence".[219]
Shallowing refers to reducing a pre-trained DNN to a smaller network with the same or similar performance.[221] Training of DNN with further shallowing can produce more efficient systems than just training of smaller networks from scratch. Shallowing is the rebirth of pruning that developed in the 1980-1990s.[222][223] The main approach to pruning is to gradually remove network elements (synapses, neurons, blocks of neurons, or layers) that have little impact on performance evaluation. Various indicators of sensitivity are used that estimate the changes of performance after pruning. The simplest indicators use just values of transmitted signals and the synaptic weights (the zeroth order). More complex indicators use mean absolute values of partial derivatives of the cost function,[223][224] 
or even the second derivatives.[222] The shallowing allows to reduce the necessary resources and makes the skills of neural network more explicit.[224] It is used for image classification,[225] for development of security systems,[226] for accelerating DNN execution on mobile devices,[227] and for other applications. It has been demonstrated that using linear postprocessing, such as supervised PCA, improves DNN performance after shallowing.[226]
qXn1  Supervised learning is the machine learning task of learning a function that maps an input to an output based on example input-output pairs.[1] It infers a function from labeled training data consisting of a set of training examples.[2]  In supervised learning, each example is a pair consisting of an input object (typically a vector) and a desired output value (also called the supervisory signal).  A supervised learning algorithm analyzes the training data and produces an inferred function, which can be used for mapping new examples. An optimal scenario will allow for the algorithm to correctly determine the class labels for unseen instances. This requires the learning algorithm to generalize from the training data to unseen situations in a "reasonable" way (see inductive bias).
The parallel task in human and animal psychology is often referred to as concept learning.
In order to solve a given problem of supervised learning, one has to perform the following steps:
A wide range of supervised learning algorithms are available, each with its strengths and weaknesses. There is no single learning algorithm that works best on all supervised learning problems (see the No free lunch theorem).
There are four major issues to consider in supervised learning:
A first issue is the tradeoff between bias and variance.[3]  Imagine that we have available several different, but equally good, training data sets.  A learning algorithm is biased for a particular input 



x


{\displaystyle x}

 if, when trained on each of these data sets, it is systematically incorrect when predicting the correct output for 



x


{\displaystyle x}

.  A learning algorithm has high variance for a particular input 



x


{\displaystyle x}

 if it predicts different output values when trained on different training sets.  The prediction error of a learned classifier is related to the sum of the bias and the variance of the learning algorithm.[4]  Generally, there is a tradeoff between bias and variance.  A learning algorithm with low bias must be "flexible" so that it can fit the data well.  But if the learning algorithm is too flexible, it will fit each training data set differently, and hence have high variance.  A key aspect of many supervised learning methods is that they are able to adjust this tradeoff between bias and variance (either automatically or by providing a bias/variance parameter that the user can adjust).
The second issue is the amount of training data available relative to the complexity of the "true" function (classifier or regression function).  If the true function is simple, then an "inflexible" learning algorithm with high bias and low variance will be able to learn it from a small amount of data.  But if the true function is highly complex (e.g., because it involves complex interactions among many different input features and behaves differently in different parts of the input space), then the function will only be able to learn from a very large amount of training data and using a "flexible" learning algorithm with low bias and high variance.
A third issue is the dimensionality of the input space.  If the input feature vectors have very high dimension, the learning problem can be difficult even if the true function only depends on a small number of those features.  This is because the many "extra" dimensions can confuse the learning algorithm and cause it to have high variance.  Hence, high input dimensional typically requires tuning the classifier to have low variance and high bias.  In practice, if the engineer can manually remove irrelevant features from the input data, this is likely to improve the accuracy of the learned function.  In addition, there are many algorithms for feature selection that seek to identify the relevant features and discard the irrelevant ones.  This is an instance of the more general strategy of dimensionality reduction, which seeks to map the input data into a lower-dimensional space prior to running the supervised learning algorithm.
A fourth issue is the degree of noise in the desired output values (the supervisory target variables).  If the desired output values are often incorrect (because of human error or sensor errors), then the learning algorithm should not attempt to find a function that exactly matches the training examples.  Attempting to fit the data too carefully leads to overfitting.  You can overfit even when there are no measurement errors (stochastic noise) if the function you are trying to learn is too complex for your learning model. In such a situation, the part of the target function that cannot be modeled "corrupts" your training data - this phenomenon has been called deterministic noise. When either type of noise is present, it is better to go with a higher bias, lower variance estimator.
In practice, there are several approaches to alleviate noise in the output values such as early stopping to prevent overfitting as well as detecting and removing the noisy training examples prior to training the supervised learning algorithm.  There are several algorithms that identify noisy training examples and removing the suspected noisy training examples prior to training has decreased generalization error with statistical significance.[5][6]
Other factors to consider when choosing and applying a learning algorithm include the following:
When considering a new application, the engineer can compare multiple learning algorithms and experimentally determine which one works best on the problem at hand (see cross validation).  Tuning the performance of a learning algorithm can be very time-consuming.  Given fixed resources, it is often better to spend more time collecting additional training data and more informative features than it is to spend extra time tuning the learning algorithms.
The most widely used learning algorithms are: 
Given a set of 



N


{\displaystyle N}

 training examples of the form 



{
(

x

1


,

y

1


)
,
.
.
.
,
(

x

N


,


y

N


)
}


{\displaystyle \{(x_{1},y_{1}),...,(x_{N},\;y_{N})\}}

 such that 




x

i




{\displaystyle x_{i}}

 is the feature vector of the i-th example and 




y

i




{\displaystyle y_{i}}

 is its label (i.e., class), a learning algorithm seeks a function 



g
:
X
→
Y


{\displaystyle g:X\to Y}

, where 



X


{\displaystyle X}

 is the input space and




Y


{\displaystyle Y}

 is the output space.  The function 



g


{\displaystyle g}

 is an element of some space of possible functions 



G


{\displaystyle G}

, usually called the hypothesis space.  It is sometimes convenient to
represent 



g


{\displaystyle g}

 using a scoring function 



f
:
X
×
Y
→

R



{\displaystyle f:X\times Y\to \mathbb {R} }

 such that 



g


{\displaystyle g}

 is defined as returning the 



y


{\displaystyle y}

 value that gives the highest score: 



g
(
x
)
=



arg
⁡
max

y



f
(
x
,
y
)


{\displaystyle g(x)={\underset {y}{\arg \max }}\;f(x,y)}

.  Let 



F


{\displaystyle F}

 denote the space of scoring functions.
Although 



G


{\displaystyle G}

 and 



F


{\displaystyle F}

 can be any space of functions, many learning algorithms are probabilistic models where 



g


{\displaystyle g}

 takes the form of a conditional probability model 



g
(
x
)
=
P
(
y

|

x
)


{\displaystyle g(x)=P(y|x)}

, or 



f


{\displaystyle f}

 takes the form of a joint probability model 



f
(
x
,
y
)
=
P
(
x
,
y
)


{\displaystyle f(x,y)=P(x,y)}

.  For example, naive Bayes and linear discriminant analysis are joint probability models, whereas logistic regression is a conditional probability model.
There are two basic approaches to choosing 



f


{\displaystyle f}

 or 



g


{\displaystyle g}

: empirical risk minimization and structural risk minimization.[7]  Empirical risk minimization seeks the function that best fits the training data.  Structural risk minimization includes a penalty function that controls the bias/variance tradeoff.
In both cases, it is assumed that the training set consists of a sample of independent and identically distributed pairs, 



(

x

i


,


y

i


)


{\displaystyle (x_{i},\;y_{i})}

.  In order to measure how well a function fits the training data, a loss function 



L
:
Y
×
Y
→


R


≥
0




{\displaystyle L:Y\times Y\to \mathbb {R} ^{\geq 0}}

 is defined.  For training example 



(

x

i


,


y

i


)


{\displaystyle (x_{i},\;y_{i})}

, the loss of predicting the value 






y
^





{\displaystyle {\hat {y}}}

 is 



L
(

y

i


,



y
^



)


{\displaystyle L(y_{i},{\hat {y}})}

.
The risk 



R
(
g
)


{\displaystyle R(g)}

 of function 



g


{\displaystyle g}

 is defined as the expected loss of 



g


{\displaystyle g}

.  This can be estimated from the training data as
In empirical risk minimization, the supervised learning algorithm seeks the function 



g


{\displaystyle g}

 that minimizes 



R
(
g
)


{\displaystyle R(g)}

.  Hence, a supervised learning algorithm can be constructed by applying an optimization algorithm to find 



g


{\displaystyle g}

.
When 



g


{\displaystyle g}

 is a conditional probability distribution 



P
(
y

|

x
)


{\displaystyle P(y|x)}

 and the loss function is the negative log likelihood: 



L
(
y
,



y
^



)
=
−
log
⁡
P
(
y

|

x
)


{\displaystyle L(y,{\hat {y}})=-\log P(y|x)}

, then empirical risk minimization is equivalent to maximum likelihood estimation.
When 



G


{\displaystyle G}

 contains many candidate functions or the training set is not sufficiently large, empirical risk minimization leads to high variance and poor generalization.  The learning algorithm is able
to memorize the training examples without generalizing well.  This is called overfitting.
Structural risk minimization seeks to prevent overfitting by incorporating a regularization penalty into the optimization.  The regularization penalty can be viewed as implementing a form of Occam's razor that prefers simpler functions over more complex ones.
A wide variety of penalties have been employed that correspond to different definitions of complexity.  For example, consider the case where the function 



g


{\displaystyle g}

 is a linear function of the form
A popular regularization penalty is 




∑

j



β

j


2




{\displaystyle \sum _{j}\beta _{j}^{2}}

, which is the squared Euclidean norm of the weights, also known as the 




L

2




{\displaystyle L_{2}}

 norm.  Other norms include the 




L

1




{\displaystyle L_{1}}

 norm, 




∑

j



|


β

j



|



{\displaystyle \sum _{j}|\beta _{j}|}

, and the 




L

0




{\displaystyle L_{0}}

 norm, which is the number of non-zero  




β

j




{\displaystyle \beta _{j}}

s.  The penalty will be denoted by 



C
(
g
)


{\displaystyle C(g)}

.
The supervised learning optimization problem is to find the function 



g


{\displaystyle g}

 that minimizes
The parameter 



λ


{\displaystyle \lambda }

 controls the bias-variance tradeoff.  When 



λ
=
0


{\displaystyle \lambda =0}

, this gives empirical risk minimization with low bias and high variance.  When 



λ


{\displaystyle \lambda }

 is large, the learning algorithm will have high bias and low variance.  The value of 



λ


{\displaystyle \lambda }

 can be chosen empirically via cross validation.
The complexity penalty has a Bayesian interpretation as the negative log prior probability of 



g


{\displaystyle g}

, 



−
log
⁡
P
(
g
)


{\displaystyle -\log P(g)}

, in which case 



J
(
g
)


{\displaystyle J(g)}

 is the posterior probabability of 



g


{\displaystyle g}

.
The training methods described above are discriminative training methods, because they seek to find a function 



g


{\displaystyle g}

 that discriminates well between the different output values (see discriminative model).  For the special case where 



f
(
x
,
y
)
=
P
(
x
,
y
)


{\displaystyle f(x,y)=P(x,y)}

 is a joint probability distribution and the loss function is the negative log likelihood 



−

∑

i


log
⁡
P
(

x

i


,

y

i


)
,


{\displaystyle -\sum _{i}\log P(x_{i},y_{i}),}

 a risk minimization algorithm is said to perform generative training, because 



f


{\displaystyle f}

 can be regarded as a generative model that explains how the data were generated.  Generative training algorithms are often simpler and more computationally efficient than discriminative training algorithms.  In some cases, the solution can be computed in closed form as in naive Bayes and linear discriminant analysis.
There are several ways in which the standard supervised learning problem can be generalized:
qXj  There is, alas, no such thing as a free lunch. This simple and obvious truth is invariably forgotten whenever irrational exuberance teams up with digital technology in the latest quest to “change the world”. A case in point was the bitcoin frenzy, where one could apparently become insanely rich by “mining” for the elusive coins. All you needed was to get a computer to solve a complicated mathematical puzzle and – lo! – you could earn one bitcoin, which at the height of the frenzy was worth $19,783.06. All you had to do was buy a mining kit (or three) from Amazon, plug it in and become part of the crypto future. The only problem was that mining became progressively more difficult the closer we got to the maximum number of bitcoins set by the scheme and so more and more computing power was required. Which meant that increasing amounts of electrical power were needed to drive the kit. Exactly how much is difficult to calculate, but one estimate published in July by the Judge Business School at the University of Cambridge suggested that the global bitcoin network was then consuming more than seven gigwatts of electricity. Over a year, that’s equal to around 64 terawatt-hours (TWh), which is 8 TWh more than Switzerland uses annually. So each of those magical virtual coins turns out to have a heavy environmental footprint. At the moment, much of the tech world is caught up in a new bout of irrational exuberance. This time, it’s about machine learning, another one of those magical technologies that “change the world”, in this case by transforming data (often obtained by spying on humans) into – depending on whom you talk to – information, knowledge and/or massive revenues. As is customary in these frenzies, some inconvenient truths are overlooked, for example, warnings by leaders in the field such as Ali Rahimi and James Mickens that the technology bears some resemblances to an older speciality called alchemy. But that’s par for the course: when you’ve embarked on changing the world (and making a fortune in the process), why let pedantic reservations get in the way? Recently, though, a newer fly has arrived in the machine-learning ointment. In a way, it’s the bitcoin problem redux. OpenAI, the San Francisco-based AI research lab, has been trying to track the amount of computing power required for machine learning ever since the field could be said to have started in 1959. What it’s found is that the history divides into two eras. From the earliest days to 2012, the amount of computing power required by the technology doubled every two years – in other words, it tracked Moore’s law of growth in processor power. But from 2012 onwards, the curve rockets upwards: the computing power required for today’s most-vaunted machine-learning systems has been doubling every 3.4 months. When you’ve embarked on changing the world, why let pedantic reservations get in the way? This hasn’t been noticed because the outfits paying the bills are huge tech companies. But the planet will notice, because the correspondingly enormous growth in electricity consumption has environmental consequences. To put that in context, researchers at Nvidia, the company that makes the specialised GPU processors now used in most machine-learning systems, came up with a massive natural-language model that was 24 times bigger than its predecessor and yet was only 34% better at its learning task. But here’s the really interesting bit. Training the final model took 512 V100 GPUs running continuously for 9.2 days. “Given the power requirements per card,” wrote one expert, “a back of the envelope estimate put the amount of energy used to train this model at over 3x the yearly energy consumption of the average American.” You don’t have to be Einstein to realise that machine learning can’t continue on its present path, especially given the industry’s frenetic assurances that tech giants are heading for an “AI everywhere” future. Brute-force cloud computing won’t achieve that goal. Of course smarter algorithms will make machine learning more resource-efficient (and perhaps also less environmentally damaging). Companies will learn to make trade-offs between accuracy and computational efficiency, though that will have unintended, and antisocial, consequences too. And, in the end, if machine learning is going to be deployed at a global scale, most of the computation will have to be done in users’ hands, ie in their smartphones. This is not as far-fetched as it sounds. The new iPhone 11, for example, includes Apple’s A13 chip, which incorporates a unit running the kind of neural network software behind recent advances in natural language processing language and interpreting images. No doubt other manufacturers have equivalent kit. In preparation for the great day of AI Everywhere, I just asked Siri: “Is there such a thing as a free lunch?” She replied: “I can help you find a restaurant if you turn on location services.” Clearly, the news that there is no such thing hasn’t yet reached Silicon Valley. They’ll get it eventually, though, when Palo Alto is underwater. Capital ideaThe Museum of Neoliberalism has just opened in Lewisham, London. It’s a wonderful project and website – my only complaint is that neoliberalism isn’t dead yet. Who needs humans?This Marketing Blog Does Not Exist is a blog entirely created by AI. Could you tell the difference between it and a human-created one? Not sure I could. All the right notesThere’s a lovely post about Handel by Ellen T Harris on the Bank of England’s blog, Bank Underground. The German composer was a shrewd investor, but it was The Messiah that made him rich.qXz�  Topics
Collective intelligence
Collective action
Self-organized criticality
Herd mentality
Phase transition
Agent-based modelling
Synchronization
Ant colony optimization
Particle swarm optimization
Swarm behaviour
Social network analysis
Small-world networks
Community identification
Centrality
Motifs
Graph Theory
Scaling
Robustness
Systems biology
Dynamic networks
Evolutionary computation
Genetic algorithms
Genetic programming
Artificial life
Machine learning
Evolutionary developmental biology
Artificial intelligence
Evolutionary robotics
Reaction–diffusion systems
Partial differential equations
Dissipative structures
Percolation
Cellular automata
Spatial ecology
Self-replication
Spatial evolutionary biology
Operationalization
Feedback
Self-reference
Goal-oriented
System dynamics
Sensemaking
Entropy
Cybernetics
Autopoiesis
Information theory
Computation theory
Ordinary differential equations
Iterative maps
Phase space
Attractors
Stability analysis
Population dynamics
Chaos
Multistability
Bifurcation
Rational choice theory
Bounded rationality
Irrational behaviour

Artificial neural networks (ANN) or connectionist systems are computing systems vaguely inspired by the biological neural networks that constitute animal brains.[1] Such systems "learn" to perform tasks by considering examples, generally without being programmed with task-specific rules. For example, in image recognition, they might learn to identify images that contain cats by analyzing example images that have been manually labeled as "cat" or "no cat" and using the results to identify cats in other images. They do this without any prior knowledge of cats, for example, that they have fur, tails, whiskers and cat-like faces. Instead, they automatically generate identifying characteristics from the examples that they process.
An ANN is based on a collection of connected units or nodes called artificial neurons, which loosely model the neurons in a biological brain. Each connection, like the synapses in a biological brain, can transmit a signal to other neurons. An artificial neuron that receives a signal then processes it and can signal neurons connected to it.
In ANN implementations, the "signal" at a connection is a real number, and the output of each neuron is computed by some non-linear function of the sum of its inputs. The connections are called edges. Neurons and edges typically have a weight that adjusts as learning proceeds. The weight increases or decreases the strength of the signal at a connection. Neurons may have a threshold such that a signal is sent only if the aggregate signal crosses that threshold. Typically, neurons are aggregated into layers. Different layers may perform different transformations on their inputs. Signals travel from the first layer (the input layer), to the last layer (the output layer), possibly after traversing the layers multiple times.
The original goal of the ANN approach was to solve problems in the same way that a human brain would. But over time, attention moved to performing specific tasks, leading to deviations from biology. ANNs have been used on a variety of tasks, including computer vision, speech recognition, machine translation, social network filtering, playing board and video games, medical diagnosis, and even in activities that have traditionally been considered as reserved to humans, like painting.[2]
Warren McCulloch and Walter Pitts[3] (1943) opened the subject by creating a computational model for neural networks.[4] In the late 1940s, D. O. Hebb[5] created a learning hypothesis based on the mechanism of neural plasticity that became known as Hebbian learning. Farley and Wesley A. Clark[6] (1954) first used computational machines, then called "calculators", to simulate a Hebbian network. Rosenblatt[7] (1958) created the perceptron.[8] The first functional networks with many layers were published by Ivakhnenko and Lapa in 1965, as the Group Method of Data Handling.[9][10][11] The basics of continuous backpropagation[9][12][13][14]  were derived in the context of control theory by Kelley[15] in 1960 and by Bryson in 1961,[16] using principles of dynamic programming.
In 1970, Seppo Linnainmaa published the general method for automatic differentiation (AD) of discrete connected networks of nested differentiable functions.[17][18] In 1973, Dreyfus used backpropagation to adapt parameters of controllers in proportion to error gradients.[19] Werbos's (1975) backpropagation algorithm enabled practical training of multi-layer networks. In 1982, he applied Linnainmaa's AD method to neural networks in the way that became widely used.[12][20] Thereafter research stagnated following Minsky and Papert (1969),[21] who discovered that basic perceptrons were incapable of processing the exclusive-or circuit and that computers lacked sufficient power to process useful neural networks. In 1992, max-pooling was introduced to help with least-shift invariance and tolerance to deformation to aid 3D object recognition.[22][23][24] Schmidhuber adopted a multi-level hierarchy of networks (1992) pre-trained one level at a time by unsupervised learning and fine-tuned by backpropagation.[25]
The development of metal–oxide–semiconductor (MOS) very-large-scale integration (VLSI), in the form of complementary MOS (CMOS) technology, enabled the development of practical artificial neural networks in the 1980s. A landmark publication in the field was the 1989 book Analog VLSI Implementation of Neural Systems by Carver A. Mead and Mohammed Ismail.[26]
Geoffrey Hinton et al. (2006) proposed learning a high-level representation using successive layers of binary or real-valued latent variables with a restricted Boltzmann machine[27] to model each layer. In 2012, Ng and Dean created a network that learned to recognize higher-level concepts, such as cats, only from watching unlabeled images.[28] Unsupervised pre-training and increased computing power from GPUs and distributed computing allowed the use of larger networks, particularly in image and visual recognition problems, which became known as "deep learning".[citation needed]
Ciresan and colleagues (2010)[29] showed that despite the vanishing gradient problem, GPUs make backpropagation feasible for many-layered feedforward neural networks.[30] Between 2009 and 2012, ANNs began winning prizes in ANN contests, approaching human level performance on various tasks, initially in pattern recognition and machine learning.[31][32] For example, the bi-directional and multi-dimensional long short-term memory (LSTM)[33][34][35][36] of Graves et al. won three competitions in connected handwriting recognition in 2009 without any prior knowledge about the three languages to be learned.[35][34]
Ciresan and colleagues built the first pattern recognizers to achieve human-competitive/superhuman performance[37] on benchmarks such as traffic sign recognition (IJCNN 2012).
ANNs began as an attempt to exploit the architecture of the human brain to perform tasks that conventional algorithms had little success with. They soon reoriented towards improving empirical results, mostly abandoning attempts to remain true to their biological precursors. Neurons are connected to each other in various patterns, to allow the output of some neurons to become the input of others. The network forms a directed, weighted graph.[38]
An artificial neural network consists of a collection of simulated neurons. Each neuron is a node which is connected to other nodes via links that correspond to biological axon-synapse-dendrite connections. Each link has a weight, which determines the strength of one node's influence on another.[39]
ANNs are composed of artificial neurons which retain the biological concept of neurons, which receive input, combine the input with their internal state (activation) and an optional threshold using an activation function, and produce output using an output function. The initial inputs are external data, such as images and documents. The ultimate outputs accomplish the task, such as recognizing an object in an image. The important characteristic of the activation function is that it provides a smooth, differentiable transition as input values change, i.e. a small change in input produces a small change in output.[40]
The network consists of connections, each connection providing the output of one neuron as an input to another neuron. Each connection is assigned a weight that represents its relative importance.[38] A given neuron can have multiple input and output connections.[41]
The propagation function computes the input to a neuron from the outputs of its predecessor neurons and their connections as a weighted sum.[38] A bias term can be added to the result of the propagation.[42]
The neurons are typically organized into multiple layers, especially in deep learning. Neurons of one layer connect only to neurons of the immediately preceding and immediately following layers. The layer that receives external data is the input layer. The layer that produces the ultimate result is the output layer. In between them are zero or more hidden layers. Single layer and unlayered networks are also used. Between two layers, multiple connection patterns are possible. They can be fully connected, with every neuron in one layer connecting to every neuron in the next layer. They can be pooling, where a group of neurons in one layer connect to a single neuron in the next layer, thereby reducing the number of neurons in that layer.[43] Neurons with only such connections form a directed acyclic graph and are known as feedforward networks.[44] Alternatively, networks that allow connections between neurons in the same or previous layers are known as recurrent networks.[45]
A hyperparameter is a constant parameter whose value is set before the learning process begins. The values of parameters are derived via learning. Examples of hyperparameters include learning rate, the number of hidden layers and batch size.[46] The values of some hyperparameters can be dependent on those of other hyperparameters. For example, the size of some layers can depend on the overall number of layers.
Learning is the adaptation of the network to better handle a task by considering sample observations. Learning involves adjusting the weights (and optional thresholds) of the network to improve the accuracy of the result. This is done by minimizing the observed errors. Learning is complete when examining additional observations does not usefully reduce the error rate. Even after learning, the error rate typically does not reach 0. If after learning, the error rate is too high, the network typically must be redesigned. Practically this is done by defining a cost function that is evaluated periodically during learning. As long as its output continues to decline, learning continues. The cost is frequently defined as a statistic whose value can only be approximated. The outputs are actually numbers, so when the error is low, the difference between the output (almost certainly a cat) and the correct answer (cat) is small. Learning attempts to reduce the total of the differences across the observations.[38] Most learning models can be viewed as a straightforward application of optimization theory and statistical estimation.
The learning rate defines the size of the corrective steps that the model takes to adjust for errors in each observation. A high learning rate shortens the training time, but with lower ultimate accuracy, while a lower learning rate takes longer, but with the potential for greater accuracy. Optimizations such as Quickprop are primarily aimed at speeding up error minimization, while other improvements mainly try to increase reliability. In order to avoid oscillation inside the network such as alternating connection weights, and to improve the rate of convergence, refinements use an adaptive learning rate that increases or decreases as appropriate.[47] The concept of momentum allows the balance between the gradient and the previous change to be weighted such that the weight adjustment depends to some degree on the previous change. A momentum close to 0 emphasizes the gradient, while a value close to 1 emphasizes the last change.
While it is possible to define a cost function  ad hoc, frequently the choice is determined by the functions desirable properties (such as convexity) or because it arises from the model (e.g., in a probabilistic model the model's posterior probability can be used as an inverse cost).
Backpropagation is a method to adjust the connection weights to compensate for each error found during learning. The error amount is effectively divided among the connections. Technically, backprop calculates the gradient (the derivative) of the cost function associated with a given state with respect to the weights. The weight updates can be done via stochastic gradient descent or other methods, such as Extreme Learning Machines,[48] "No-prop" networks,[49] training without backtracking,[50] "weightless" networks,[51][52] and non-connectionist neural networks.
The three major learning paradigms are supervised learning, unsupervised learning and reinforcement learning. They each correspond to a particular learning task
Supervised learning uses a set of paired inputs and desired outputs. The learning task is to produce the desired output for each input. In this case the cost function is related to eliminating incorrect deductions.[53] A commonly used cost is the mean-squared error, which tries to minimize the average squared error between the network's output and the desired output. Tasks suited for supervised learning are pattern recognition (also known as classification) and regression (also known as function approximation). Supervised learning is also applicable to sequential data (e.g., for hand writing, speech and gesture recognition). This can be thought of as learning with a "teacher", in the form of a function that provides continuous feedback on the quality of solutions obtained thus far.
In unsupervised learning, input data is given along with the cost function, some function of the data 




x



{\displaystyle \textstyle x}

 and the network's output. The cost function is dependent on the task (the model domain) and any a priori assumptions (the implicit properties of the model, its parameters and the observed variables). As a trivial example, consider the model 




f
(
x
)
=
a



{\displaystyle \textstyle f(x)=a}

 where 




a



{\displaystyle \textstyle a}

 is a constant and the cost 




C
=
E
[
(
x
−
f
(
x
)

)

2


]



{\displaystyle \textstyle C=E[(x-f(x))^{2}]}

. Minimizing this cost produces a value of 




a



{\displaystyle \textstyle a}

 that is equal to the mean of the data. The cost function can be much more complicated. Its form depends on the application: for example, in compression it could be related to the mutual information between 




x



{\displaystyle \textstyle x}

 and 




f
(
x
)



{\displaystyle \textstyle f(x)}

, whereas in statistical modeling, it could be related to the posterior probability of the model given the data (note that in both of those examples those quantities would be maximized rather than minimized). Tasks that fall within the paradigm of unsupervised learning are in general estimation problems; the applications include clustering, the estimation of statistical distributions, compression and filtering.
In applications such as playing video games, an actor takes a string of actions, receiving a generally unpredictable response from the environment after each one. The goal is to win the game, i.e., generate the most positive (lowest cost) responses. In reinforcement learning, the aim is to weight the network (devise a policy) to perform actions that minimize long-term (expected cumulative) cost. At each point in time the agent performs an action and the environment generates an observation and an instantaneous cost, according to some (usually unknown) rules. The rules and the long-term cost usually only can be estimated. At any juncture, the agent decides whether to explore new actions to uncover their costs or to exploit prior learning to proceed more quickly.
Formally the environment is modeled as a Markov decision process (MDP) with states 






s

1


,
.
.
.
,

s

n



∈
S



{\displaystyle \textstyle {s_{1},...,s_{n}}\in S}

 and actions 






a

1


,
.
.
.
,

a

m



∈
A



{\displaystyle \textstyle {a_{1},...,a_{m}}\in A}

. Because the state transitions are not known, probability distributions are used instead: the instantaneous cost distribution 




P
(

c

t



|


s

t


)



{\displaystyle \textstyle P(c_{t}|s_{t})}

, the observation distribution 




P
(

x

t



|


s

t


)



{\displaystyle \textstyle P(x_{t}|s_{t})}

 and the transition distribution 




P
(

s

t
+
1



|


s

t


,

a

t


)



{\displaystyle \textstyle P(s_{t+1}|s_{t},a_{t})}

, while a policy is defined as the conditional distribution over actions given the observations. Taken together, the two define a Markov chain (MC). The aim is to discover the lowest-cost MC.
ANNs serve as the learning component in such applications.[54][55] Dynamic programming coupled with ANNs (giving neurodynamic programming)[56] has been applied to problems such as those involved in vehicle routing,[57] video games, natural resource management[58][59] and medicine[60] because of ANNs ability to mitigate losses of accuracy even when reducing the discretization grid density for numerically approximating the solution of control problems. Tasks that fall within the paradigm of reinforcement learning are control problems, games and other sequential decision making tasks.
Self learning in neural networks was introduced in 1982 along with a neural network capable of self-learning  named Crossbar Adaptive Array (CAA).[61] It is a system with only one input, situation s, and only one output, action (or behavior) a. It has neither external advice input nor external reinforcement input from the environment. The CAA computes, in a crossbar fashion, both decisions about actions and emotions (feelings) about encountered situations. The system is driven by the interaction between cognition and emotion.[62] Given memory matrix W =||w(a,s)||, the crossbar self learning algorithm in each iteration performs the following computation:
The backpropagated value (secondary reinforcement) is the emotion toward the consequence situation. The CAA exists in two environments, one is behavioral environment where it behaves, and the other is genetic environment, where from it initially and only once receives initial emotions about to be encountered situations in the behavioral environment. Having received the genome vector (species vector) from the genetic environment, the CAA will learn a goal-seeking behavior, in the behavioral environment that contains both desirable and undesirable situations.[63]
In a Bayesian framework, a distribution over the set of allowed models is chosen to minimize the cost. Evolutionary methods,[64] gene expression programming,[65] simulated annealing,[66] expectation-maximization, non-parametric methods and particle swarm optimization[67] are other learning algorithms. Convergent recursion is a learning algorithm for cerebellar model articulation controller (CMAC) neural networks.[68][69]
Two modes of learning are available: stochastic and batch. In stochastic learning, each input creates a weight adjustment. In batch learning weights are adjusted based on a batch of inputs, accumulating errors over the batch. Stochastic learning introduces "noise" into the process, using the local gradient calculated from one data point; this reduces the chance of the network getting stuck in local minima. However, batch learning typically yields a faster, more stable descent to a local minimum, since each update is performed in the direction of the batch's average error. A common compromise is to use "mini-batches", small batches with samples in each batch selected stochastically from the entire data set.
ANNs have evolved into a broad family of techniques that have advanced the state of the art across multiple domains.  The simplest types have one or more static components, including number of units, number of layers, unit weights and topology. Dynamic types allow one or more of these to evolve via learning. The latter are much more complicated, but can shorten learning periods and produce better results. Some types allow/require learning to be "supervised" by the operator, while others operate independently. Some types operate purely in hardware, while others are purely software and run on general purpose computers.
Some of the main breakthroughs include: convolutional neural networks that have proven particularly successful in processing visual and other two-dimensional data;[70][71] long short-term memory avoid the vanishing gradient problem[72] and can handle signals that have a mix of low and high frequency components aiding large-vocabulary speech recognition,[73][74] text-to-speech synthesis,[75][12][76] and photo-real talking heads;[77] competitive networks such as generative adversarial networks in which multiple networks (of varying structure) compete with each other, on tasks such as winning a game[78] or on deceiving the opponent about the authenticity of an input.[79]
Neural architecture search (NAS) uses machine learning to automate ANN design. Various approaches to NAS have designed networks that compare well with hand-designed systems. The basic search algorithm is to propose a candidate model, evaluate it against a dataset and use the results as feedback to teach the NAS network.[80] Available systems include AutoML and AutoKeras.[81]
Design issues include deciding the number, type and connectedness of network layers, as well as the size of each and the connection type (full, pooling, ...).
Hyperparameters must also be defined as part of the design (they are not learned), governing matters such as how many neurons are in each layer, learning rate, step, stride, depth, receptive field and padding (for CNNs), etc.[82]
Using Artificial neural networks requires an understanding of their characteristics.
ANN capabilities fall within the following broad categories:[citation needed]
Because of their ability to reproduce and model nonlinear processes, Artificial neural networks have found applications in many disciplines. Application areas include system identification and control (vehicle control, trajectory prediction,[83] process control, natural resource management), quantum chemistry,[84] general game playing,[85] pattern recognition (radar systems, face identification, signal classification,[86] 3D reconstruction,[87] object recognition and more), sequence recognition (gesture, speech, handwritten and printed text recognition), medical diagnosis, finance[88] (e.g. automated trading systems), data mining, visualization, machine translation, social network filtering[89] and e-mail spam filtering. ANNs have been used to diagnose cancers, including lung cancer,[90] prostate cancer, colorectal cancer[91] and to distinguish highly invasive cancer cell lines from less invasive lines using only cell shape information.[92][93]
ANNs have been used to accelerate reliability analysis of infrastructures subject to natural disasters[94][95] and to predict foundation settlements.[96] ANNs have also been used for building black-box models in geoscience: hydrology,[97][98] ocean modelling and coastal engineering,[99][100] and geomorphology.[101] ANNs have been employed in cybersecurity, with the objective to discriminate between legitimate activities and malicious ones. For example, machine learning has been used for classifying Android malware,[102] for identifying domains belonging to threat actors and for detecting URLs posing a security risk.[103] Research is underway on ANN systems designed for penetration testing, for detecting botnets,[104] credit cards frauds[105] and network intrusions.
ANNs have been proposed as a tool to simulate the properties of many-body open quantum systems.[106][107][108][109] In brain research ANNs have studied short-term behavior of individual neurons,[110] the dynamics of neural circuitry arise from interactions between individual neurons and how behavior can arise from abstract neural modules that represent complete subsystems. Studies considered long-and short-term plasticity of neural systems and their relation to learning and memory from the individual neuron to the system level.
The multilayer perceptron is a universal function approximator, as proven by the universal approximation theorem. However, the proof is not constructive regarding the number of neurons required, the network topology, the weights and the learning parameters.
A specific recurrent architecture with rational-valued weights (as opposed to full precision real number-valued weights) has the power of a universal Turing machine,[111] using a finite number of neurons and standard linear connections. Further, the use of irrational values for weights results in a machine with super-Turing power.[112]
A model's "capacity" property roughly corresponds to its ability to model any given function. It is related to the amount of information that can be stored in the network and to the notion of complexity.[citation needed]
Models may not consistently converge on a single solution, firstly because local minima may exist, depending on the cost function and the model. Secondly, the optimization method used might not guarantee to converge when it begins far from any local minimum. Thirdly, for sufficiently large data or parameters, some methods become impractical.
Convergence behavior of certain types of ANN architectures are more understood than others. Such as when the width of network approaches to infinity, the ANN resembles linear model, thus such ANN follows the convergence behavior of linear model also.[113] Another example is when parameters are small, it is observed that  ANN often fits target functions from low to high frequencies.[114][115][116][117] Such phenomenon is in opposite to the behavior of some well studied iterative numerical schemes such as Jacobi method.
Applications whose goal is to create a system that generalizes well to unseen examples, face the possibility of over-training. This arises in convoluted or over-specified systems when the network capacity significantly exceeds the needed free parameters. Two approaches address over-training. The first is to use cross-validation and similar techniques to check for the presence of over-training and to select hyperparameters to minimize the generalization error.
The second is to use some form of regularization. This concept emerges in a probabilistic (Bayesian) framework, where regularization can be performed by selecting a larger prior probability over simpler models; but also in statistical learning theory, where the goal is to minimize over two quantities: the 'empirical risk' and the 'structural risk', which roughly corresponds to the error over the training set and the predicted error in unseen data due to overfitting.
Supervised neural networks that use a mean squared error (MSE) cost function can use formal statistical methods to determine the confidence of the trained model. The MSE on a validation set can be used as an estimate for variance. This value can then be used to calculate the confidence interval of network output, assuming a normal distribution. A confidence analysis made this way is statistically valid as long as the output probability distribution stays the same and the network is not modified.
By assigning a softmax activation function, a generalization of the logistic function, on the output layer of the neural network (or a softmax component in a component-based network) for categorical target variables, the outputs can be interpreted as posterior probabilities. This is useful in classification as it gives a certainty measure on classifications.
The softmax activation function is:

A common criticism of neural networks, particularly in robotics, is that they require too much training for real-world operation.[citation needed] Potential solutions include randomly shuffling training examples, by using a numerical optimization algorithm that does not take too large steps when changing the network connections following an example, grouping examples in so-called mini-batches and/or introducing a recursive least squares algorithm for CMAC.[68]
A fundamental objection is that ANNs do not sufficiently reflect neuronal function. Backpropagation is a critical step, although no such mechanism exists in biological neural networks.[118] How information is coded by real neurons is not known. Sensor neurons fire action potentials more frequently with sensor activation and muscle cells pull more strongly when their associated motor neurons receive action potentials more frequently.[119] Other than the case of relaying information from a sensor neuron to a motor neuron, almost nothing of the principles of how information is handled by biological neural networks is known.
A central claim of ANNs is that they embody new and powerful general principles for processing information. Unfortunately, these principles are ill-defined. It is often claimed that they are emergent from the network itself. This allows simple statistical association (the basic function of artificial neural networks) to be described as learning or recognition. Alexander Dewdney commented that, as a result, artificial neural networks have a "something-for-nothing quality, one that imparts a peculiar aura of laziness and a distinct lack of curiosity about just how good these computing systems are. No human hand (or mind) intervenes; solutions are found as if by magic; and no one, it seems, has learned anything".[120] One response to Dewdney is that neural networks handle many complex and diverse tasks, ranging from autonomously flying aircraft[121] to detecting credit card fraud to mastering the game of Go.
Technology writer Roger Bridgman commented:
Neural networks, for instance, are in the dock not only because they have been hyped to high heaven, (what hasn't?) but also because you could create a successful net without understanding how it worked: the bunch of numbers that captures its behaviour would in all probability be "an opaque, unreadable table...valueless as a scientific resource".
In spite of his emphatic declaration that science is not technology, Dewdney seems here to pillory neural nets as bad science when most of those devising them are just trying to be good engineers. An unreadable table that a useful machine could read would still be well worth having.[122]
Biological brains use both shallow and deep circuits as reported by brain anatomy,[123] displaying a wide variety of invariance. Weng[124] argued that the brain self-wires largely according to signal statistics and therefore, a serial cascade cannot catch all major statistical dependencies.
Large and effective neural networks require considerable computing resources.[125] While the brain has hardware tailored to the task of processing signals through a graph of neurons, simulating even a simplified neuron on von Neumann architecture may consume vast amounts of memory and storage. Furthermore, the designer often needs to transmit signals through many of these connections and their associated neurons –  which require enormous CPU power and time.
Schmidhuber noted that the resurgence of neural networks in the twenty-first century is largely attributable to advances in hardware: from 1991 to 2015, computing power, especially as delivered by GPGPUs (on GPUs), has increased around a million-fold, making the standard backpropagation algorithm feasible for training networks that are several layers deeper than before.[126] The use of accelerators such as FPGAs and GPUs can reduce training times from months to days.[127][125]
Neuromorphic engineering addresses the hardware difficulty directly, by constructing non-von-Neumann chips to directly implement neural networks in circuitry. Another type of chip optimized for neural network processing is called a Tensor Processing Unit, or TPU.[128]
Analyzing what has been learned by an ANN, is much easier than to analyze what has been learned by a biological neural network. Furthermore, researchers involved in exploring learning algorithms for neural networks are gradually uncovering general principles that allow a learning machine to be successful. For example, local vs. non-local learning and shallow vs. deep architecture.[129]
Advocates of hybrid models (combining neural networks and symbolic approaches), claim that such a mixture can better capture the mechanisms of the human mind.[130][131]
A single-layer feedforward artificial neural network. Arrows originating from 





x

2





{\displaystyle \scriptstyle x_{2}}

 are omitted for clarity. There are p inputs to this network and q outputs. In this system, the value of the qth output, 





y

q





{\displaystyle \scriptstyle y_{q}}

 would be calculated as 





y

q


=
K
∗
(
∑
(

x

i


∗

w

i
q


)
−

b

q


)



{\displaystyle \scriptstyle y_{q}=K*(\sum (x_{i}*w_{iq})-b_{q})}


A two-layer feedforward artificial neural network.
An artificial neural network.
An ANN dependency graph.
A single-layer feedforward artificial neural network with 4 inputs, 6 hidden and 2 outputs. Given position state and direction outputs wheel based control values.
A two-layer feedforward artificial neural network with 8 inputs, 2x8 hidden and 2 outputs. Given position state, direction and other environment values outputs thruster based control values.
Parallel pipeline structure of CMAC neural network. This learning algorithm can converge in one step.
qX�/  Decision tree learning is one of the predictive modeling approaches used in statistics, data mining and machine learning. It uses a decision tree (as a predictive model) to go from observations about an item (represented in the branches) to conclusions about the item's target value (represented in the leaves). Tree models where the target variable can take a discrete set of values are called classification trees; in these tree structures, leaves represent class labels and branches represent conjunctions of features that lead to those class labels. Decision trees where the target variable can take continuous values (typically real numbers) are called regression trees.
In decision analysis, a decision tree can be used to visually and explicitly represent decisions and decision making. In data mining, a decision tree describes data (but the resulting classification tree can be an input for decision making). This page deals with decision trees in data mining.
Decision tree learning is a method commonly used in data mining.[1] The goal is to create a model that predicts the value of a target variable based on several input variables. 
A decision tree is a simple representation for classifying examples. For this section, assume that all of the input features have finite discrete domains, and there is a single target feature called the "classification". Each element of the domain of the classification is called a class.
A decision tree or a classification tree is a tree in which each internal (non-leaf) node is labeled with an input feature. The arcs coming from a node labeled with an input feature are labeled with each of the possible values of the target or output feature or the arc leads to a subordinate decision node on a different input feature. Each leaf of the tree is labeled with a class or a probability distribution over the classes, signifying that the data set has been classified by the tree into either a specific class, or into a particular probability distribution (which, if the decision tree is well-constructed, is skewed towards certain subsets of classes).
A tree is built by splitting the source set, constituting the root node of the tree, into subsets - which constitute the successor children. The splitting is based on a set of splitting rules based on classification features.[2]  This process is repeated on each derived subset in a recursive manner called recursive partitioning.
The recursion is completed when the subset at a node has all the same values of the target variable, or when splitting no longer adds value to the predictions. This process of top-down induction of decision trees (TDIDT)[3] is an example of a greedy algorithm, and it is by far the most common strategy for learning decision trees from data[citation needed].
In data mining, decision trees can be described also as the combination of mathematical and computational techniques to aid the description, categorization and generalization of a given set of data.
Data comes in records of the form:
The dependent variable, 



Y


{\displaystyle Y}

, is the target variable that we are trying to understand, classify or generalize. The vector 





x




{\displaystyle {\textbf {x}}}

 is composed of the features, 




x

1


,

x

2


,

x

3




{\displaystyle x_{1},x_{2},x_{3}}

 etc., that are used for that task.
Decision trees used in data mining are of two main types:
The term Classification And Regression Tree (CART) analysis is an umbrella term used to refer to both of the above procedures, first introduced by Breiman et al. in 1984.[4] Trees used for regression and trees used for classification have some similarities - but also some differences, such as the procedure used to determine where to split.[4]
Some techniques, often called ensemble methods, construct more than one decision tree:
A special case of a decision tree is a decision list,[9] which is a one-sided decision tree, so that every internal node has exactly 1 leaf node and exactly 1 internal node as a child (except for the bottommost node, whose only child is a single leaf node).  While less expressive, decision lists are arguably easier to understand than general decision trees due to their added sparsity, permit non-greedy learning methods[10] and monotonic constraints to be imposed.[11]
Decision tree learning is the construction of a decision tree from class-labeled training tuples. A decision tree is a flow-chart-like structure, where each internal (non-leaf) node denotes a test on an attribute, each branch represents the outcome of a test, and each leaf (or terminal) node holds a class label. The topmost node in a tree is the root node.
There are many specific decision-tree algorithms. Notable ones include:
ID3 and CART were invented independently at around the same time (between 1970 and 1980)[citation needed], yet follow a similar approach for learning a decision tree from training tuples.
Algorithms for constructing decision trees usually work top-down, by choosing a variable at each step that best splits the set of items.[15] Different algorithms use different metrics for measuring "best".  These generally measure the homogeneity of the target variable within the subsets. Some examples are given below. These metrics are applied to each candidate subset, and the resulting values are combined (e.g., averaged) to provide a measure of the quality of the split.
Used by the CART (classification and regression tree) algorithm for classification trees, Gini impurity is a measure of how often a randomly chosen element from the set would be incorrectly labeled if it was randomly labeled according to the distribution of labels in the subset. The Gini impurity can be computed by summing the probability 




p

i




{\displaystyle p_{i}}

 of an item with label 



i


{\displaystyle i}

  being chosen times the probability 




∑

k
≠
i



p

k


=
1
−

p

i




{\displaystyle \sum _{k\neq i}p_{k}=1-p_{i}}

 of a mistake in categorizing that item.  It reaches its minimum (zero) when all cases in the node fall into a single target category. 
The Gini impurity is also an information theoretic measure and corresponds to Tsallis Entropy with deformation coefficient 



q
=
2


{\displaystyle q=2}

, which in Physics is associated with the lack of information in out-of-equlibrium, non-extensive, dissipative and quantum systems. For the limit 



q
→
1


{\displaystyle q\to 1}

 one recovers the usual Boltzmann-Gibbs or Shannon entropy. In this sense, the Gini impurity is but a variation of the usual entropy measure for decision trees.
To compute Gini impurity for a set of items with 



J


{\displaystyle J}

 classes, suppose 



i
∈
{
1
,
2
,
.
.
.
,
J
}


{\displaystyle i\in \{1,2,...,J\}}

, and let 




p

i




{\displaystyle p_{i}}

 be the fraction of items labeled with class 



i


{\displaystyle i}

 in the set.
Used by the ID3, C4.5 and C5.0 tree-generation algorithms. Information gain is based on the concept of entropy and information content from information theory.
Entropy is defined as below
where 




p

1


,

p

2


,
.
.
.


{\displaystyle p_{1},p_{2},...}

are fractions that add up to 1 and represent the percentage of each class present in the child node that results from a split in the tree.[16]
Information gain is used to decide which feature to split on at each step in building the tree. Simplicity is best, so we want to keep our tree small. To do so, at each step we should choose the split that results in the purest daughter nodes. A commonly used measure of purity is called information which is measured in bits. For each node of the tree, the information value "represents the expected amount of information that would be needed to specify whether a new instance should be classified yes or no, given that the example reached that node".[16]
Consider an example data set with four attributes: outlook (sunny, overcast, rainy), temperature (hot, mild, cool), humidity (high, normal), and windy (true, false), with a binary (yes or no) target variable, play, and 14 data points. To construct a decision tree on this data, we need to compare the information gain of each of four trees, each split on one of the four features. The split with the highest information gain will be taken as the first split and the process will continue until all children nodes are pure, or until the information gain is 0.
The split using the feature windy results in two children nodes, one for a windy value of true and one for a windy value of false. In this data set, there are six data points with a true windy value, three of which have a play (where play is the target variable) value of yes and three with a play value of no. The eight remaining data points with a windy value of false contain two no's and six yes's. The information of the windy=true node is calculated using the entropy equation above. Since there is an equal number of yes's and no's in this node, we have
For the node where windy=false there were eight data points, six yes's and two no's. Thus we have
To find the information of the split, we take the weighted average of these two numbers based on how many observations fell into which node.
To find the information gain of the split using windy, we must first calculate the information in the data before the split. The original data contained nine yes's and five no's.
Now we can calculate the information gain achieved by splitting on the windy feature.
To build the tree, the information gain of each possible first split would need to be calculated. The best first split is the one that provides the most information gain. This process is repeated for each impure node until the tree is complete. This example is adapted from the example appearing in Witten et al.[16]
Introduced in CART,[4] variance reduction is often employed in cases where the target variable is continuous (regression tree), meaning that use of many other metrics would first require discretization before being applied. The variance reduction of a node N is defined as the total reduction of the variance of the target variable x due to the split at this node:
where 



S


{\displaystyle S}

, 




S

t




{\displaystyle S_{t}}

, and 




S

f




{\displaystyle S_{f}}

 are the set of presplit sample indices, set of sample indices for which the split test is true, and set of sample indices for which the split test is false, respectively. Each of the above summands are indeed variance estimates, though, written in a form without directly referring to the mean.
Amongst other data mining methods, decision trees have various advantages:
Many data mining software packages provide implementations of one or more decision tree algorithms.
Examples include Salford Systems CART (which licensed the proprietary code of the original CART authors),[4] IBM SPSS Modeler, RapidMiner, SAS Enterprise Miner, Matlab, R (an open-source software environment for statistical computing, which includes several CART implementations such as rpart, party and randomForest packages), Weka (a free and open-source data-mining suite, contains many decision tree algorithms), Orange, KNIME, Microsoft SQL Server [1], and scikit-learn (a free and open-source machine learning library for the Python programming language).
In a decision tree, all paths from the root node to the leaf node proceed by way of conjunction, or AND. In a decision graph, it is possible to use disjunctions (ORs) to join two more paths together using minimum message length (MML).[26]  Decision graphs have been further extended to allow for previously unstated new attributes to be learnt dynamically and used at different places within the graph.[27]  The more general coding scheme results in better predictive accuracy and log-loss probabilistic scoring.[citation needed]  In general, decision graphs infer models with fewer leaves than decision trees.
Evolutionary algorithms have been used to avoid local optimal decisions and search the decision tree space with little a priori bias.[28][29]
It is also possible for a tree to be sampled using MCMC.[30]
The tree can be searched for in a bottom-up fashion.[31]
q	X~  In machine learning, a hyperparameter is a parameter whose value is set before the learning process begins. By contrast, the values of other parameters are derived via training.
Hyperparameters can be classified as model hyperparameters, that cannot be inferred while fitting the machine to the training set because they refer to the model selection task, or algorithm hyperparameters, that in principle have no influence on the performance of the model but affect the speed and quality of the learning process. An example of the first type is the topology and size of a neural network. An example of the second type is learning rate or mini-batch size.
Different model training algorithms require different hyperparameters, some simple algorithms (such as ordinary least squares regression) require none. Given these hyperparameters, the training algorithm learns the parameters from the data. For instance, LASSO is an algorithm that adds a regularization hyperparameter to ordinary least squares regression, which has to be set before estimating the parameters through the training algorithm.
The time required to train and test a model can depend upon the choice of its hyperparameters.[1] A hyperparameter is usually of continuous or integer type, leading to mixed-type optimization problems.[1] The existence of some hyperparameters is conditional upon the value of others, e.g. the size of each hidden layer in a neural network can be conditional upon the number of layers.[1]
Usually, but not always, hyperparameters cannot be learned using well known gradient based methods (such as gradient descent, LBFGS) - which are commonly employed to learn parameters. These hyperparameters are those parameters describing a model representation that cannot be learned by common optimization methods but nonetheless affect the loss function. An example would be the tolerance hyperparameter for errors in support vector machines.
Sometimes, hyperparameters cannot be learned from the training data because they aggressively increase the capacity of a model and can push the loss function to a bad minimum - overfitting to, and picking up noise, in the data - as opposed to correctly mapping the richness of the structure in the data. For example - if we treat the degree of a polynomial equation fitting a regression model as a trainable parameter - this would just raise the degree up until the model perfectly fit the data, giving small training error - but bad generalization performance.
Most performance variation can be attributed to just a few hyperparameters.[2][1][3] The tunability of an algorithm, hyperparameter, or interacting hyperparameters is a measure of how much performance can be gained by tuning it.[4] For an LSTM, while the learning rate followed by the network size are its most crucial hyperparameters,[5] batching and momentum have no significant effect on its performance.[6]
Although some research has advocated the use of mini-batch sizes in the thousands, other work has found the best performance with mini-batch sizes between 2 and 32.[7]
An inherent stochasticity in learning directly implies that the empirical hyperparameter performance is not necessarily its true performance.[1] Methods that are not robust to simple changes in hyperparameters, random seeds, or even different implementations of the same algorithm cannot be integrated into mission critical control systems without significant simplification and robustification.[8]
Reinforcement learning algorithms, in particular, require measuring their performance over a large number of random seeds, and also measuring their sensitivity to choices of hyperparameters.[8] Their evaluation with a small number of random seeds does not capture performance adequately due to high variance.[8] Some reinforcement learning methods, e.g. DDPG (Deep Deterministic Policy Gradient), are more sensitive to hyperparameter choices than others.[8]
Hyperparameter optimization finds a tuple of hyperparameters that yields an optimal model which minimizes a predefined loss function on given test data.[1]  The objective function takes a tuple of hyperparameters and returns the associated loss.[1]
Apart from tuning hyperparameters, machine learning involves storing and organizing the parameters and results, and making sure they are reproducible.[9] In the absence of a robust infrastructure for this purpose, research code often evolves quickly and compromises essential aspects like bookkeeping and reproducibility.[10] Online collaboration platforms for machine learning go further by allowing scientists to automatically share, organize and discuss experiments, data, and algorithms.[11]
A number of relevant services and open source software exist:
q
XU  The Guardian is right to express legitimate concerns about the opacity of machine learning systems and attempts to replicate what humans do best (Editorial, 23 September), and we welcome this. However, as founders of the Institute for Ethical AI in Education (IEAIED) we believe these problems must be overcome in order to ensure people are able to benefit from artificial intelligence, not just fear it. There are highly beneficial applications of machine learning. In education, for example, this innovation will enable personalised learning for all and is already enabling individualised learning support for increasing numbers of students. Well-designed AI can be used to identify learners’ particular needs so that everyone – especially the most vulnerable – can receive targeted support. Given the magnitude of what people have to gain from machine learning tools, we feel an obligation to mitigate and counteract the inherent risks so that the best possible outcomes can be realised. First, we must not accept that machine learning systems have to be block-boxes whose decisions and behaviours are beyond the reach of human understanding. Explainable AI (XAI) is a rapidly developing field, and we encourage education stakeholders to demand and expect high levels of transparency. There are also further means by which we can ethically derive benefits from machine learning systems, while retaining human responsibility. Another approach to benefiting from AI without being undermined by a lack of human oversight is to consider that AI is not bringing about these benefits single-handedly. Genuine advancement arises when AI augments and assists human-driven processes and skills. Machine learning is a powerful tool for informing strategy and decision-making, but people remain responsible for how that information is harnessed. Incorporating ethics into the design and development of AI-driven technology is vital, and we currently rely on programmes such as UCL Educate, an accelerator for education SMEs and startups, to instil that ethos in innovation from the concept stage. Crucially, though, we must inform the public at large about AI – what it is and what benefits can be derived from its use – or we risk alienating people from the technology that already forms part of their everyday lives. Worse still, we risk causing alarm and making them fearful.Prof Rose Luckin Professor of learner centred design at UCL Institute of Education and director of UCL Educate Sir Anthony Seldon Vice-chancellor, University of Buckingham Priya Lakhani Founder CEO, Century Tech • Join the debate – email guardian.letters@theguardian.com • Read more Guardian letters – click here to visit gu.com/letters • Do you have a photo you’d like to share with Guardian readers? Click here to upload it and we’ll publish the best submissions in the letters spread of our print editionqX�(  In statistics, overfitting is "the production of an analysis that corresponds too closely or exactly to a particular set of data, and may therefore fail to fit additional data or predict future observations reliably".[1] An overfitted model is a statistical model that contains more parameters than can be justified by the data.[2] The essence of overfitting is to have unknowingly extracted some of the residual variation (i.e. the noise) as if that variation represented underlying model structure.[3]:45
Underfitting occurs when a statistical model cannot adequately capture the underlying structure of the data. An underfitted model is a model where some parameters or terms that would appear in a correctly specified model are missing.[2] Underfitting would occur, for example, when fitting a linear model to non-linear data. Such a model will tend to have poor predictive performance.
Overfitting and underfitting can occur in machine learning, in particular. In machine learning, the phenomena are sometimes called "overtraining" and "undertraining". 
The possibility of overfitting exists because the criterion used for selecting the model is not the same as the criterion used to judge the suitability of a model. For example, a model might be selected by maximizing its performance on some set of training data, and yet its suitability might be determined by its ability to perform well on unseen data; then overfitting occurs when a model begins to "memorize" training data rather than "learning" to generalize from a trend. 
As an extreme example, if the number of parameters is the same as or greater than the number of observations, then a model can perfectly predict the training data simply by memorizing the data in its entirety. (For an illustration, see Figure 2.) Such a model, though, will typically fail severely when making predictions. 
The potential for overfitting depends not only on the number of parameters and data but also the conformability of the model structure with the data shape, and the magnitude of model error compared to the expected level of noise or error in the data.[citation needed] Even when the fitted model does not have an excessive number of parameters, it is to be expected that the fitted relationship will appear to perform less well on a new data set than on the data set used for fitting (a phenomenon sometimes known as shrinkage).[2] In particular, the value of the coefficient of determination will shrink relative to the original data.
To lessen the chance of, or amount of, overfitting, several techniques are available (e.g. model comparison, cross-validation, regularization, early stopping, pruning, Bayesian priors, or dropout). The basis of some techniques is either (1) to explicitly penalize overly complex models or (2) to test the model's ability to generalize by evaluating its performance on a set of data not used for training, which is assumed to approximate the typical unseen data that a model will encounter.
In statistics, an inference is drawn from a statistical model, which has been selected via some procedure. Burnham & Anderson, in their much-cited text on model selection, argue that to avoid overfitting, we should adhere to the "Principle of Parsimony".[3] The authors also state the following.[3]:32–33
Overfitted models … are often free of bias in the parameter estimators, but have estimated (and actual) sampling variances that are needlessly large (the precision of the estimators is poor, relative to what could have been accomplished with a more parsimonious model). False treatment effects tend to be identified, and false variables are included with overfitted models. … A best approximating model is achieved by properly balancing the errors of underfitting and overfitting.Overfitting is more likely to be a serious concern when there is little theory available to guide the analysis, in part because then there tend to be a large number of models to select from. The book Model Selection and Model Averaging (2008) puts it this way.[4]
Given a data set, you can fit thousands of models at the push of a button, but how do you choose the best? With so many candidate models, overfitting is a real danger. Is the monkey who typed Hamlet actually a good writer?In regression analysis, overfitting occurs frequently.[5] As an extreme example, if there are p variables in a linear regression with p data points, the fitted line can go exactly through every point.[6] For logistic regression or Cox proportional hazards models, there are a variety of rules of thumb (e.g. 5–9[7], 10[8] and 10–15[9] — the guideline of 10 observations per independent variable is known as the "one in ten rule"). In the process of regression model selection, the mean squared error of the random regression function can be split into random noise, approximation bias, and variance in the estimate of the regression function. The bias–variance tradeoff is often used to overcome overfit models.
With a large set of explanatory variables that actually have no relation to the dependent variable being predicted, some variables will in general be falsely found to be statistically significant and the researcher may thus retain them in the model, thereby overfitting the model. This is known as Freedman's paradox.
Usually a learning algorithm is trained using some set of "training data": exemplary situations for which the desired output is known. The goal is that the algorithm will also perform well on predicting the output when fed "validation data" that was not encountered during its training.
Overfitting is the use of models or procedures that violate Occam's razor, for example by including more adjustable parameters than are ultimately optimal, or by using a more complicated approach than is ultimately optimal. For an example where there are too many adjustable parameters, consider a dataset where training data for y can be adequately predicted by a linear function of two dependent variables. Such a function requires only three parameters (the intercept and two slopes). Replacing this simple function with a new, more complex quadratic function, or with a new, more complex linear function on more than two dependent variables, carries a risk: Occam's razor implies that any given complex function is a priori less probable than any given simple function. If the new, more complicated function is selected instead of the simple function, and if there was not a large enough gain in training-data fit to offset the complexity increase, then the new complex function "overfits" the data, and the complex overfitted function will likely perform worse than the simpler function on validation data outside the training dataset, even though the complex function performed as well, or perhaps even better, on the training dataset.[10]
When comparing different types of models, complexity cannot be measured solely by counting how many parameters exist in each model; the expressivity of each parameter must be considered as well. For example, it is nontrivial to directly compare the complexity of a neural net (which can track curvilinear relationships) with m parameters to a regression model with n parameters.[10]
Overfitting is especially likely in cases where learning was performed too long or where training examples are rare, causing the learner to adjust to very specific random features of the training data that have no causal relation to the target function. In this process of overfitting, the performance on the training examples still increases while the performance on unseen data becomes worse.
As a simple example, consider a database of retail purchases that includes the item bought, the purchaser, and the date and time of purchase. It's easy to construct a model that will fit the training set perfectly by using the date and time of purchase to predict the other attributes, but this model will not generalize at all to new data, because those past times will never occur again.
Generally, a learning algorithm is said to overfit relative to a simpler one if it is more accurate in fitting known data (hindsight) but less accurate in predicting new data (foresight). One can intuitively understand overfitting from the fact that information from all past experience can be divided into two groups: information that is relevant for the future, and irrelevant information ("noise"). Everything else being equal, the more difficult a criterion is to predict (i.e., the higher its uncertainty), the more noise exists in past information that needs to be ignored. The problem is determining which part to ignore. A learning algorithm that can reduce the chance of fitting noise is called "robust."
The most obvious consequence of overfitting is poor performance on the validation dataset. Other negative consequences include:[10]
The optimal function usually needs verification on bigger or completely new datasets.  There are, however, methods like minimum spanning tree or life-time of correlation that applies the dependence between correlation coefficients and time-series (window width). Whenever the window width is big enough, the correlation coefficients are stable and don't depend on the window width size anymore. Therefore, a correlation matrix can be created by calculating a coefficient of correlation between investigated variables. This matrix can be represented topologically as a complex network where direct and indirect influences between variables are visualized.
Underfitting occurs when a statistical model or machine learning algorithm cannot adequately capture the underlying structure of the data. It occurs when the model or algorithm does not fit the data enough. Underfitting occurs if the model or algorithm shows low variance but high bias (to contrast the opposite, overfitting from high variance and low bias). It is often a result of an excessively simple model.[11]
Burnham & Anderson state the following.[3]:32
… an underfitted model would ignore some important replicable (i.e., conceptually replicable in most other samples) structure in the data and thus fail to identify effects that were actually supported by the data. In this case, bias in the parameter estimators is often substantial, and the sampling variance is underestimated, both factors resulting in poor confidence interval coverage. Underfitted models tend to miss important treatment effects in experimental settings.qX$  In machine learning and statistics, classification is the problem of identifying to which of a set of categories (sub-populations) a new observation belongs, on the basis of a training set of data containing observations (or instances) whose category membership is known.  Examples are assigning a given email to the "spam" or "non-spam" class, and assigning a diagnosis to a given patient based on observed characteristics of the patient (sex, blood pressure, presence or absence of certain symptoms, etc.).  Classification is an example of pattern recognition.
In the terminology of machine learning,[1] classification is considered an instance of supervised learning, i.e., learning where a training set of correctly identified observations is available.  The corresponding unsupervised procedure is known as clustering, and involves grouping data into categories based on some measure of inherent similarity or distance.
Often, the individual observations are analyzed into a set of quantifiable properties, known variously as explanatory variables or features.  These properties may variously be categorical (e.g. "A", "B", "AB" or "O", for blood type), ordinal (e.g. "large", "medium" or "small"), integer-valued (e.g. the number of occurrences of a particular word in an email) or real-valued (e.g. a measurement of blood pressure). Other classifiers work by comparing observations to previous observations by means of a similarity or distance function.
An algorithm that implements classification, especially in a concrete implementation, is known as a classifier.  The term "classifier" sometimes also refers to the mathematical function, implemented by a classification algorithm, that maps input data to a category.
Terminology across fields is quite varied. In statistics, where classification is often done with logistic regression or a similar procedure, the properties of observations are termed explanatory variables (or independent variables, regressors, etc.), and the categories to be predicted are known as outcomes, which are considered to be possible values of the dependent variable.  In machine learning, the observations are often known as instances, the explanatory variables are termed features (grouped into a feature vector), and the possible categories to be predicted are classes.  Other fields may use different terminology: e.g. in community ecology, the term "classification" normally refers to cluster analysis, i.e., a type of unsupervised learning, rather than the supervised learning described in this article.
Classification and clustering are examples of the more general problem of pattern recognition, which is the assignment of some sort of output value to a given input value.  Other examples are regression, which assigns a real-valued output to each input; sequence labeling, which assigns a class to each member of a sequence of values (for example, part of speech tagging, which assigns a part of speech to each word in an input sentence); parsing, which assigns a parse tree to an input sentence, describing the syntactic structure of the sentence; etc.
A common subclass of classification is probabilistic classification.  Algorithms of this nature use statistical inference to find the best class for a given instance.  Unlike other algorithms, which simply output a "best" class, probabilistic algorithms output a probability of the instance being a member of each of the possible classes.  The best class is normally then selected as the one with the highest probability.  However, such an algorithm has numerous advantages over non-probabilistic classifiers:
Early work on statistical classification was undertaken by Fisher,[2][3] in the context of two-group problems, leading to Fisher's linear discriminant function as the rule for assigning a group to a new observation.[4] This early work assumed that data-values within each of the two groups had a multivariate normal distribution. The extension of this same context to more than two-groups has also been considered with a restriction imposed that the classification rule should be linear.[4][5] Later work for the multivariate normal distribution allowed the classifier to be nonlinear:[6] several classification rules can be derived based on different adjustments of the Mahalanobis distance, with a new observation being assigned to the group whose centre has the lowest adjusted distance from the observation.
Unlike frequentist procedures, Bayesian classification procedures provide a natural way of taking into account any available information about the relative sizes of the different groups within the overall population.[7] Bayesian procedures tend to be computationally expensive and, in the days before Markov chain Monte Carlo computations were developed, approximations for Bayesian clustering rules were devised.[8]
Some Bayesian procedures involve the calculation of  group membership probabilities: these can be viewed as providing a more informative outcome of a data analysis than a simple attribution of a single group-label to each new observation.
Classification can be thought of as two separate problems – binary classification and multiclass classification. In binary classification, a better understood task, only two classes are involved, whereas multiclass classification involves assigning an object to one of several classes.[9] Since many classification methods have been developed specifically for binary classification, multiclass classification often requires the combined use of multiple binary classifiers.
Most algorithms describe an individual instance whose category is to be predicted using a feature vector of individual, measurable properties of the instance.  Each property is termed a feature, also known in statistics as an explanatory variable (or independent variable, although features may or may not be statistically independent).  Features may variously be binary (e.g. "on" or "off"); categorical (e.g. "A", "B", "AB" or "O", for blood type); ordinal (e.g. "large", "medium" or "small"); integer-valued (e.g. the number of occurrences of a particular word in an email); or real-valued (e.g. a measurement of blood pressure).  If the instance is an image, the feature values might correspond to the pixels of an image; if the instance is a piece of text, the feature values might be occurrence frequencies of different words.  Some algorithms work only in terms of discrete data and require that real-valued or integer-valued data be discretized into groups (e.g. less than 5, between 5 and 10, or greater than 10).
A large number of algorithms for classification can be phrased in terms of a linear function that assigns a score to each possible category k by combining the feature vector of an instance with a vector of weights, using a dot product.  The predicted category is the one with the highest score.  This type of score function is known as a linear predictor function and has the following general form:
where Xi is the feature vector for instance i, βk is the vector of weights corresponding to category k, and score(Xi, k) is the score associated with assigning instance i to category k.  In discrete choice theory, where instances represent people and categories represent choices, the score is considered the utility associated with person i choosing category k.
Algorithms with this basic setup are known as linear classifiers.  What distinguishes them is the procedure for determining (training) the optimal weights/coefficients and the way that the score is interpreted.
Examples of such algorithms are
In unsupervised learning, classifiers form the backbone of cluster analysis and in supervised or semi-supervised learning, classifiers are how the system characterizes and evaluates unlabeled data. In all cases though, classifiers have a specific set of dynamic rules, which includes an interpretation procedure to handle vague or unknown values, all tailored to the type of inputs being examined.[10]
Since no single form of classification is appropriate for all data sets, a large toolkit of classification algorithms have been developed. The most commonly used include:[11]
Classifier performance depends greatly on the characteristics of the data to be classified. There is no single classifier that works best on all given problems (a phenomenon that may be explained by the no-free-lunch theorem). Various empirical tests have been performed to compare classifier performance and to find the characteristics of data that determine classifier performance. Determining a suitable classifier for a given problem is however still more an art than a science.
The measures precision and recall are popular metrics used to evaluate the quality of a classification system. More recently, receiver operating characteristic (ROC) curves have been used to evaluate the tradeoff between true- and false-positive rates of classification algorithms.
As a performance metric, the uncertainty coefficient has the advantage over simple accuracy in that it is not affected by the relative sizes of the different classes.
[12]
Further, it will not penalize an algorithm for simply rearranging the classes.
Classification has many applications. In some of these it is employed as a data mining procedure, while in others more detailed statistical modeling is undertaken.
qXzJ  Pattern recognition is the automated recognition of patterns and regularities in data. Pattern recognition is closely related to artificial intelligence and machine learning,[1] together with applications such as data mining and knowledge discovery in databases (KDD), and is often used interchangeably with these terms. However, these are distinguished: machine learning is one approach to pattern recognition, while other approaches include hand-crafted (not learned) rules or heuristics; and pattern recognition is one approach to artificial intelligence, while other approaches include symbolic artificial intelligence.[2] A modern definition of pattern recognition is:
The field of pattern recognition is concerned with the automatic discovery of regularities in data through the use of computer algorithms and with the use of these regularities to take actions such as classifying the data into different categories.[3]This article focuses on machine learning approaches to pattern recognition. Pattern recognition systems are in many cases trained from labeled "training" data (supervised learning), but when no labeled data are available other algorithms can be used to discover previously unknown patterns (unsupervised learning). Machine learning is strongly related to pattern recognition and originates from artificial intelligence. KDD and data mining have a larger focus on unsupervised methods and stronger connection to business use. Pattern recognition focuses more on the signal and also takes acquisition and Signal Processing into consideration. It originated in engineering, and the term is popular in the context of computer vision: a leading computer vision conference is named Conference on Computer Vision and Pattern Recognition. In pattern recognition, there may be a higher interest to formalize, explain and visualize the pattern, while machine learning traditionally focuses on maximizing the recognition rates. Yet, all of these domains have evolved substantially from their roots in artificial intelligence, engineering and statistics, and they've become increasingly similar by integrating developments and ideas from each other.
In machine learning, pattern recognition is the assignment of a label to a given input value. In statistics, discriminant analysis was introduced for this same purpose in 1936. An example of pattern recognition is classification, which attempts to assign each input value to one of a given set of classes (for example, determine whether a given email is "spam" or "non-spam"). However, pattern recognition is a more general problem that encompasses other types of output as well. Other examples are regression, which assigns a real-valued output to each input;[4] sequence labeling, which assigns a class to each member of a sequence of values [5](for example, part of speech tagging, which assigns a part of speech to each word in an input sentence); and parsing, which assigns a parse tree to an input sentence, describing the syntactic structure of the sentence.[6]
Pattern recognition algorithms generally aim to provide a reasonable answer for all possible inputs and to perform "most likely" matching of the inputs, taking into account their statistical variation. This is opposed to pattern matching algorithms, which look for exact matches in the input with pre-existing patterns. A common example of a pattern-matching algorithm is regular expression matching, which looks for patterns of a given sort in textual data and is included in the search capabilities of many text editors and word processors. In contrast to pattern recognition, pattern matching is not generally a type of machine learning, although pattern-matching algorithms (especially with fairly general, carefully tailored patterns) can sometimes succeed in providing similar-quality output of the sort provided by pattern-recognition algorithms.
Pattern recognition is generally categorized according to the type of learning procedure used to generate the output value. Supervised learning assumes that a set of training data (the training set) has been provided, consisting of a set of instances that have been properly labeled by hand with the correct output. A learning procedure then generates a model that attempts to meet two sometimes conflicting objectives: Perform as well as possible on the training data, and generalize as well as possible to new data (usually, this means being as simple as possible, for some technical definition of "simple", in accordance with Occam's Razor, discussed below). Unsupervised learning, on the other hand, assumes training data that has not been hand-labeled, and attempts to find inherent patterns in the data that can then be used to determine the correct output value for new data instances.[7] A combination of the two that has recently been explored is semi-supervised learning, which uses a combination of labeled and unlabeled data (typically a small set of labeled data combined with a large amount of unlabeled data). Note that in cases of unsupervised learning, there may be no training data at all to speak of; in other words,and the data to be labeled is the training data.
Note that sometimes different terms are used to describe the corresponding supervised and unsupervised learning procedures for the same type of output. For example, the unsupervised equivalent of classification is normally known as clustering, based on the common perception of the task as involving no training data to speak of, and of grouping the input data into clusters based on some inherent similarity measure (e.g. the distance between instances, considered as vectors in a multi-dimensional vector space), rather than assigning each input instance into one of a set of pre-defined classes. In some fields, the terminology is different: For example, in community ecology, the term "classification" is used to refer to what is commonly known as "clustering".
The piece of input data for which an output value is generated is formally termed an instance. The instance is formally described by a vector of features, which together constitute a description of all known characteristics of the instance. (These feature vectors can be seen as defining points in an appropriate multidimensional space, and methods for manipulating vectors in vector spaces can be correspondingly applied to them, such as computing the dot product or the angle between two vectors.) Typically, features are either categorical (also known as nominal, i.e., consisting of one of a set of unordered items, such as a gender of "male" or "female", or a blood type of "A", "B", "AB" or "O"), ordinal (consisting of one of a set of ordered items, e.g., "large", "medium" or "small"), integer-valued (e.g., a count of the number of occurrences of a particular word in an email) or real-valued (e.g., a measurement of blood pressure). Often, categorical and ordinal data are grouped together; likewise for integer-valued and real-valued data. Furthermore, many algorithms work only in terms of categorical data and require that real-valued or integer-valued data be discretized into groups (e.g., less than 5, between 5 and 10, or greater than 10).
Many common pattern recognition algorithms are probabilistic in nature, in that they use statistical inference to find the best label for a given instance. Unlike other algorithms, which simply output a "best" label, often probabilistic algorithms also output a probability of the instance being described by the given label. In addition, many probabilistic algorithms output a list of the N-best labels with associated probabilities, for some value of N, instead of simply a single best label. When the number of possible labels is fairly small (e.g., in the case of classification), N may be set so that the probability of all possible labels is output. Probabilistic algorithms have many advantages over non-probabilistic algorithms:
Feature selection algorithms attempt to directly prune out redundant or irrelevant features. A general introduction to feature selection which summarizes approaches and challenges, has been given.[8] The complexity of feature-selection is, because of its non-monotonous character, an optimization problem where given a total of 



n


{\displaystyle n}

 features the powerset consisting of all 




2

n


−
1


{\displaystyle 2^{n}-1}

 subsets of features need to be explored. The Branch-and-Bound algorithm[9] does reduce this complexity but is intractable for medium to large values of the number of available features 



n


{\displaystyle n}

. For a large-scale comparison of feature-selection algorithms see 
.[10]
Techniques to transform the raw feature vectors (feature extraction) are sometimes used prior to application of the pattern-matching algorithm. For example, feature extraction algorithms attempt to reduce a large-dimensionality feature vector into a smaller-dimensionality vector that is easier to work with and encodes less redundancy, using mathematical techniques such as principal components analysis (PCA). The distinction between feature selection and feature extraction is that the resulting features after feature extraction has taken place are of a different sort than the original features and may not easily be interpretable, while the features left after feature selection are simply a subset of the original features.
Formally, the problem of supervised pattern recognition can be stated as follows: Given an unknown function 



g
:


X


→


Y




{\displaystyle g:{\mathcal {X}}\rightarrow {\mathcal {Y}}}

 (the ground truth) that maps input instances 




x

∈


X




{\displaystyle {\boldsymbol {x}}\in {\mathcal {X}}}

 to output labels 



y
∈


Y




{\displaystyle y\in {\mathcal {Y}}}

, along with training data 




D

=
{
(


x


1


,

y

1


)
,
…
,
(


x


n


,

y

n


)
}


{\displaystyle \mathbf {D} =\{({\boldsymbol {x}}_{1},y_{1}),\dots ,({\boldsymbol {x}}_{n},y_{n})\}}

 assumed to represent accurate examples of the mapping, produce a function 



h
:


X


→


Y




{\displaystyle h:{\mathcal {X}}\rightarrow {\mathcal {Y}}}

 that approximates as closely as possible the correct mapping 



g


{\displaystyle g}

. (For example, if the problem is filtering spam, then 





x


i




{\displaystyle {\boldsymbol {x}}_{i}}

 is some representation of an email and 



y


{\displaystyle y}

 is either "spam" or "non-spam"). In order for this to be a well-defined problem, "approximates as closely as possible" needs to be defined rigorously. In decision theory, this is defined by specifying a loss function or cost function that assigns a specific value to "loss" resulting from producing an incorrect label. The goal then is to minimize the expected loss, with the expectation taken over the probability distribution of 





X




{\displaystyle {\mathcal {X}}}

. In practice, neither the distribution of 





X




{\displaystyle {\mathcal {X}}}

 nor the ground truth function 



g
:


X


→


Y




{\displaystyle g:{\mathcal {X}}\rightarrow {\mathcal {Y}}}

 are known exactly, but can be computed only empirically by collecting a large number of samples of 





X




{\displaystyle {\mathcal {X}}}

 and hand-labeling them using the correct value of 





Y




{\displaystyle {\mathcal {Y}}}

 (a time-consuming process, which is typically the limiting factor in the amount of data of this sort that can be collected). The particular loss function depends on the type of label being predicted. For example, in the case of classification, the simple zero-one loss function is often sufficient. This corresponds simply to assigning a loss of 1 to any incorrect labeling and implies that the optimal classifier minimizes the error rate on independent test data (i.e. counting up the fraction of instances that the learned function 



h
:


X


→


Y




{\displaystyle h:{\mathcal {X}}\rightarrow {\mathcal {Y}}}

 labels wrongly, which is equivalent to maximizing the number of correctly classified instances). The goal of the learning procedure is then to minimize the error rate (maximize the correctness) on a "typical" test set.
For a probabilistic pattern recognizer, the problem is instead to estimate the probability of each possible output label given a particular input instance, i.e., to estimate a function of the form
where the feature vector input is 




x



{\displaystyle {\boldsymbol {x}}}

, and the function f is typically parameterized by some parameters 




θ



{\displaystyle {\boldsymbol {\theta }}}

.[11] In a discriminative approach to the problem, f is estimated directly. In a generative approach, however, the inverse probability 



p
(


x


|



l
a
b
e
l



)


{\displaystyle p({{\boldsymbol {x}}|{\rm {label}}})}

 is instead estimated and combined with the prior probability 



p
(


l
a
b
e
l



|


θ

)


{\displaystyle p({\rm {label}}|{\boldsymbol {\theta }})}

 using Bayes' rule, as follows:
When the labels are continuously distributed (e.g., in regression analysis), the denominator involves integration rather than summation:
The value of 




θ



{\displaystyle {\boldsymbol {\theta }}}

 is typically learned using maximum a posteriori (MAP) estimation. This finds the best value that simultaneously meets two conflicting objects: To perform as well as possible on the training data (smallest error-rate) and to find the simplest possible model. Essentially, this combines maximum likelihood estimation with a regularization procedure that favors simpler models over more complex models. In a Bayesian context, the regularization procedure can be viewed as placing a prior probability 



p
(

θ

)


{\displaystyle p({\boldsymbol {\theta }})}

 on different values of 




θ



{\displaystyle {\boldsymbol {\theta }}}

. Mathematically:
where 





θ


∗




{\displaystyle {\boldsymbol {\theta }}^{*}}

 is the value used for 




θ



{\displaystyle {\boldsymbol {\theta }}}

 in the subsequent evaluation procedure, and 



p
(

θ


|


D

)


{\displaystyle p({\boldsymbol {\theta }}|\mathbf {D} )}

, the posterior probability of 




θ



{\displaystyle {\boldsymbol {\theta }}}

, is given by
In the Bayesian approach to this problem, instead of choosing a single parameter vector 





θ


∗




{\displaystyle {\boldsymbol {\theta }}^{*}}

, the probability of a given label for a new instance 




x



{\displaystyle {\boldsymbol {x}}}

 is computed by integrating over all possible values of 




θ



{\displaystyle {\boldsymbol {\theta }}}

, weighted according to the posterior probability:
The first pattern classifier – the linear discriminant presented by Fisher – was developed in the frequentist tradition. The frequentist approach entails that the model parameters are considered unknown, but objective. The parameters are then computed (estimated) from the collected data. For the linear discriminant, these parameters are precisely the mean vectors and the covariance matrix. Also the probability of each class 



p
(


l
a
b
e
l



|


θ

)


{\displaystyle p({\rm {label}}|{\boldsymbol {\theta }})}

 is estimated from the collected dataset. Note that the usage of 'Bayes rule' in a pattern classifier does not make the classification approach Bayesian.
Bayesian statistics has its origin in Greek philosophy where a distinction was already made between the 'a priori' and the 'a posteriori' knowledge. Later Kant defined his distinction between what is a priori known – before observation – and the empirical knowledge gained from observations. In a Bayesian pattern classifier, the class probabilities 



p
(


l
a
b
e
l



|


θ

)


{\displaystyle p({\rm {label}}|{\boldsymbol {\theta }})}

 can be chosen by the user, which are then a priori. Moreover, experience quantified as a priori parameter values can be weighted with empirical observations – using e.g., the Beta- (conjugate prior) and Dirichlet-distributions. The Bayesian approach facilitates a seamless intermixing between expert knowledge in the form of subjective probabilities, and objective observations.
Probabilistic pattern classifiers can be used according to a frequentist or a Bayesian approach.
Within medical science, pattern recognition is the basis for computer-aided diagnosis (CAD) systems. CAD describes a procedure that supports the doctor's interpretations and findings.
Other typical applications of pattern recognition techniques are automatic speech recognition, classification of text into several categories (e.g., spam/non-spam email messages), the automatic recognition of handwriting on postal envelopes, automatic recognition of images of human faces, or handwriting image extraction from medical forms.[12] The last two examples form the subtopic image analysis of pattern recognition that deals with digital images as input to pattern recognition systems.[13][14]
Optical character recognition is a classic example of the application of a pattern classifier, see
OCR-example.
The method of signing one's name was captured with stylus and overlay starting in 1990.[citation needed] The strokes, speed, relative min, relative max, acceleration and pressure is used to uniquely identify and confirm identity. Banks were first offered this technology, but were content to collect from the FDIC for any bank fraud and did not want to inconvenience customers..[citation needed]
Artificial neural networks (neural net classifiers) and deep learning have many real-world applications in image processing, a few examples:
For a discussion of the aforementioned applications of neural networks in image processing, see e.g.[24]
In psychology, pattern recognition (making sense of and identifying objects) is closely related to perception, which explains how the sensory inputs humans receive are made meaningful. Pattern recognition can be thought of in two different ways: the first being template matching and the second being feature detection. 
A template is a pattern used to produce items of the same proportions. The template-matching hypothesis suggests that incoming stimuli are compared with templates in the long term memory. If there is a match, the stimulus is identified.
Feature detection models, such as the Pandemonium system for classifying letters (Selfridge, 1959), suggest that the stimuli are broken down into their component parts for identification. For example, a capital E has three horizontal lines and one vertical line.[25]
Algorithms for pattern recognition depend on the type of label output, on whether learning is supervised or unsupervised, and on whether the algorithm is statistical or non-statistical in nature. Statistical algorithms can further be categorized as generative or discriminative.
Parametric:[26]
Nonparametric:[27]
Unsupervised:
Supervised (?):
Supervised:
Unsupervised:
Supervised:
Unsupervised:
This article is based on material taken from  the Free On-line Dictionary of Computing  prior to 1 November 2008 and incorporated under the "relicensing" terms of the GFDL, version 1.3 or later.
qX�  It turns out you don’t need to be Dr Doolittle to eavesdrop on arguments in the animal kingdom. Researchers studying Egyptian fruit bats say they have found a way to work out who is arguing with whom, what they are squabbling about and can even predict the outcome of a disagreement – all from the bats’ calls.  “The global quest is to understand where human language comes from. To do this we must study animal communication,” said Yossi Yovel, co-author of the research from Tel Aviv University in Israel. “One of the big questions in animal communication is how much information is conveyed.”  Egyptian fruit bats, common to Africa and the Middle East, are social creatures. But the calls they make as they huddle together to roost are almost impossible to tell apart by human ear, all simply sounding aggressive. “Basically [it’s] bats shouting at each other,” said Yovel. But, writing in the journal Scientific Reports, Yovel and colleagues describe how they managed to discern meaning within the squeaks.  The approach, they reveal, relied on harnessing machine learning algorithms originally used for human voice recognition. A form of artificial intelligence, machine learning algorithms are “trained” by being fed data that has already been sorted into categories, and then used to apply the patterns and relationships the system has spotted to sort new data. The team spent 75 days continuously recording both audio and video footage of 22 bats that were split into two groups and housed in separate cages. By studying the video footage, the researchers were able to unpick which bats were arguing each other, the outcome of each row, and sort the squabbles into four different bones of contention: sleep, food, perching position and unwanted mating attempts. The team then trained the machine learning algorithm with around 15,000 bat calls from seven adult females, each categorised using information gleaned from the video footage, before testing the system’s accuracy.  The results revealed that, based only on the frequencies within the bats’ calls, the algorithm correctly identified the bat making the call around 71% of the time, and what the animals were squabbling about around 61% of the time. The system was also able to identify, although with less accuracy, who the call was aimed at and predict the fallout of the disagreement, revealing whether the bats would part or not, and if so, which bat would leave.  The differences between the calls were nuanced. “What we find is there are certain pitch differences that characterise the different categories - but it is not as if you can say mating [calls] are high vocalisations and eating are low,” said Yovel. The results, he says, reveals that even everyday calls are rich in information. “We have shown that a big bulk of bat vocalisations that previously were thought to all mean the same thing, something like ‘get out of here!’ actually contain a lot of information,” said Yovel, adding that analysing further aspects of the bats’ calls, such as their patterns and stresses, could reveal even more detail encoded in the squeaks. Kate Jones, professor of ecology and biodiversity at University College, London described the findings as exciting. “It is like a Rosetta stone to getting into [the bats’] social behaviours,” she said of the team’s approach. “I really like the fact that they have managed to decode some of this vocalisation and there is much more information in these signals than we thought.” With the approach based on the social sounds made between bats, Jones says the technique could be used to shed light on how other species of animals communicate. “It could be that you could apply the same type of techniques to other species to figure out what they mean when they are interacting with each other,” she said. “So it could be that it opens up a different world of understanding what these communications are.”qX�  With machine learning behind myriad technologies, from online dating to driverless cars, we ask expert Dr Michael Osborne from the University of Oxford to give us the lowdown. What is machine learning? Artificial intelligence is really the goal of trying to develop algorithms that can learn and act. Older AI research ran up against this difficulty that people didn’t really know what intelligence was. Machine learning is a much more modern approach to solving the AI problem where we are coming from the bottom up rather than top down, so we are saying, well, let’s define these really crisp, well-defined, sub-problems, like classifying a handwritten digit as being either a one or two, and then use novel techniques within statistics and optimisation to create an algorithm that can improve its performance over time. What is an algorithm? An algorithm is a sequence of instructions or rules to complete a task. Where do we encounter machine learning? Anywhere and everywhere. The spam filter in your email account will be built around a machine learning algorithm. If you use Google translate , machine learning is trying to learn patterns in data to translate text from one language to another. Dating sites will use machine learning algorithms to try and recommend potential mates for you on the basis of your interests. [With] recommendation systems we are seeing the rise of machine learning all over the internet - if you buy products from Amazon it will recommend things to you on the basis of what you buy but also on what those like you buy. What are the benefits? It is not subject to the same biases or heuristics that humans are - for example, a study of judges found that their decisions about whether or not to award parole to prisoners were influenced by whether or not they had just had lunch. Algorithms can [also] scale up much better than humans can. You can fit many more processors than humans into a room. What are the challenges? There are technical challenges, obviously, related to designing these crisp problems that are relevant to a large section of industrial and societal needs. [Also] people are concerned that as algorithms get more intelligent they might start to substitute for what humans can do - they might impact on employment. We have this burden to try and make sure that we develop technologies that complement humans rather than replace them. What about the future? Health informatics will affect a lot of people’s lives. Everyone is carrying around phones that are recording characteristics of their activities - perhaps we can use that information to try and identify when someone’s about to have a heart attack or a seizure. Again this notion of trying to pick out underlying trends or patterns in data and use it to make predictions. The task of figuring out where a robot is in the world is very much a machine learning problem. Something I think a lot of people are justifiably excited about are autonomous vehicles – that is going to be a huge change in our lives.qXְ  
In deep learning, a convolutional neural network (CNN, or ConvNet) is a class of deep neural networks, most commonly applied to analyzing visual imagery. They are also known as shift invariant or space invariant artificial neural networks (SIANN), based on their shared-weights architecture and translation invariance characteristics.[1][2] They have applications in image and video recognition, recommender systems,[3] image classification, medical image analysis, natural language processing,[4] and financial time series.[5].mw-parser-output .toclimit-2 .toclevel-1 ul,.mw-parser-output .toclimit-3 .toclevel-2 ul,.mw-parser-output .toclimit-4 .toclevel-3 ul,.mw-parser-output .toclimit-5 .toclevel-4 ul,.mw-parser-output .toclimit-6 .toclevel-5 ul,.mw-parser-output .toclimit-7 .toclevel-6 ul{display:none}CNNs are regularized versions of multilayer perceptrons. Multilayer perceptrons usually mean fully connected networks, that is, each neuron in one layer is connected to all neurons in the next layer. The "fully-connectedness" of these networks makes them prone to overfitting data. Typical ways of regularization include adding some form of magnitude measurement of weights to the loss function. CNNs take a different approach towards regularization: they take advantage of the hierarchical pattern in data and assemble more complex patterns using smaller and simpler patterns. Therefore, on the scale of connectedness and complexity, CNNs are on the lower extreme.
Convolutional networks were inspired by biological processes[6][7][8][9] in that the connectivity pattern between neurons resembles the organization of the animal visual cortex. Individual cortical neurons respond to stimuli only in a restricted region of the visual field known as the receptive field. The receptive fields of different neurons partially overlap such that they cover the entire visual field.
CNNs use relatively little pre-processing compared to other image classification algorithms. This means that the network learns the filters that in traditional algorithms were hand-engineered. This independence from prior knowledge and human effort in feature design is a major advantage.
The name “convolutional neural
network” indicates that the network employs a mathematical operation called
convolution. Convolution is a specialized kind of linear operation. Convolutional
networks are simply neural networks that use convolution in place of general matrix
multiplication in at least one of their layers.[10]
A convolutional neural network consists of an input and an output layer, as well as multiple hidden layers. The hidden layers of a CNN typically consist of a series of convolutional layers that convolve with a multiplication or other dot product. The activation function is commonly a RELU layer, and is subsequently followed by additional convolutions such as pooling layers, fully connected layers and normalization layers, referred to as hidden layers because their inputs and outputs are masked by the activation function and final convolution.
Though the layers are colloquially referred to as convolutions, this is only by convention. Mathematically, it is technically a sliding dot product or cross-correlation. This has significance for the indices in the matrix, in that it affects how weight is determined at a specific index point.
When programming a CNN, the input is a tensor with shape (number of images) x (image width) x (image height) x (image depth). Then after passing through a convolutional layer, the image becomes abstracted to a feature map, with shape (number of images) x (feature map width) x (feature map height) x (feature map channels). A convolutional layer within a neural network should have the following attributes:
Convolutional layers convolve the input and pass its result to the next layer. This is similar to the response of a neuron in the visual cortex to a specific stimulus.[11] Each convolutional neuron processes data only for its receptive field. Although fully connected feedforward neural networks can be used to learn features as well as classify data, it is not practical to apply this architecture to images. A very high number of neurons would be necessary, even in a shallow (opposite of deep) architecture, due to the very large input sizes associated with images, where each pixel is a relevant variable. For instance, a fully connected layer for a (small) image of size 100 x 100 has 10,000 weights for each neuron in the second layer. The convolution operation brings a solution to this problem as it reduces the number of free parameters, allowing the network to be deeper with fewer parameters.[12] For instance, regardless of image size, tiling regions of size 5 x 5, each with the same shared weights, requires only 25 learnable parameters. In this way, it resolves the vanishing or exploding gradients problem in training traditional multi-layer neural networks with many layers by using backpropagation.[citation needed]
Convolutional networks may include local or global pooling layers to streamline the underlying computation. Pooling layers reduce the dimensions of the data by combining the outputs of neuron clusters at one layer into a single neuron in the next layer. Local pooling combines small clusters, typically 2 x 2. Global pooling acts on all the neurons of the convolutional layer.[13][14] In addition, pooling may compute a max or an average. Max pooling uses the maximum value from each of a cluster of neurons at the prior layer.[15][16] Average pooling uses the average value from each of a cluster of neurons at the prior layer.[17]
Fully connected layers connect every neuron in one layer to every neuron in another layer. It is in principle the same as the traditional multi-layer perceptron neural network (MLP). The flattened matrix goes through a fully connected layer to classify the images.
In neural networks, each neuron receives input from some number of locations in the previous layer. In a fully connected layer, each neuron receives input from every element of the previous layer. In a convolutional layer, neurons receive input from only a restricted subarea of the previous layer. Typically the subarea is of a square shape (e.g., size 5 by 5). The input area of a neuron is called its receptive field. So, in a fully connected layer, the receptive field is the entire previous layer. In a convolutional layer, the receptive area is smaller than the entire previous layer.
Each neuron in a neural network computes an output value by applying a specific function to the input values coming from the receptive field in the previous layer. The function that is applied to the input values is determined by a vector of weights and a bias (typically real numbers). Learning, in a neural network, progresses by making iterative adjustments to these biases and weights.
The vector of weights and the bias are called filters and represent particular features of the input (e.g., a particular shape). A distinguishing feature of CNNs is that many neurons can share the same filter. This reduces memory footprint because a single bias and a single vector of weights are used across all receptive fields sharing that filter, as opposed to each receptive field having its own bias and vector weighting.[18]
CNN design follows vision processing in living organisms.[citation needed]
Work by Hubel and Wiesel in the 1950s and 1960s showed that cat and monkey visual cortexes contain neurons that individually respond to small regions of the visual field. Provided the eyes are not moving, the region of visual space within which visual stimuli affect the firing of a single neuron is known as its receptive field.[citation needed] Neighboring cells have similar and overlapping receptive fields.[citation needed] Receptive field size and location varies systematically across the cortex to form a complete map of visual space.[citation needed] The cortex in each hemisphere represents the contralateral visual field.[citation needed]
Their 1968 paper identified two basic visual cell types in the brain:[7]
Hubel and Wiesel also proposed a cascading model of these two types of cells for use in pattern recognition tasks.[19][20]
The "neocognitron"[6] was introduced by Kunihiko Fukushima in 1980.[8][16][21]
It was inspired by the above-mentioned work of Hubel and Wiesel. The neocognitron introduced the two basic types of layers in CNNs: convolutional layers, and downsampling layers. A convolutional layer contains units whose receptive fields cover a patch of the previous layer. The weight vector (the set of adaptive parameters) of such a unit is often called a filter. Units can share filters. Downsampling layers contain units whose receptive fields cover patches of previous convolutional layers. Such a unit typically computes the average of the activations of the units in its patch. This downsampling helps to correctly classify objects in visual scenes even when the objects are shifted.
In a variant of the neocognitron called the cresceptron, instead of using Fukushima's spatial averaging, J. Weng et al. introduced a method called max-pooling where a downsampling unit computes the maximum of the activations of the units in its patch.[22] Max-pooling is often used in modern CNNs.[23]
Several supervised and unsupervised learning algorithms have been proposed over the decades to train the weights of a neocognitron.[6] Today, however, the CNN architecture is usually trained through backpropagation.
The neocognitron is the first CNN which requires units located at multiple network positions to have shared weights. Neocognitrons were adapted in 1988 to analyze time-varying signals.[24]
The time delay neural network (TDNN) was introduced in 1987 by Alex Waibel et al. and was the first convolutional network, as it achieved shift invariance.[25] It did so by utilizing weight sharing in combination with Backpropagation training.[26] Thus, while also using a pyramidal structure as in the neocognitron, it performed a global optimization of the weights, instead of a local one.[25]
TDNNs are convolutional networks that share weights along the temporal dimension.[27] They allow speech signals to be processed time-invariantly. In 1990 Hampshire and Waibel introduced a variant which performs a two dimensional convolution.[28] Since these TDNNs operated on spectrograms the resulting phoneme recognition system was invariant to both, shifts in time and in frequency. This inspired translation invariance in image processing with CNNs.[26] The tiling of neuron outputs can cover timed stages.[29]
TDNNs now achieve the best performance in far distance speech recognition.[30]
In 1990 Yamaguchi et al. introduced the concept of max pooling. They did so by combining TDNNs with max pooling in order to realize a speaker independent isolated word recognition system.[15] In their system they used several TDNNs per word, one for each syllable. The results of each TDNN over the input signal were combined using max pooling and the outputs of the pooling layers were then passed on to networks performing the actual word classification.
A system to recognize hand-written ZIP Code numbers[31] involved convolutions in which the kernel coefficients had been laboriously hand designed.[32]
Yann LeCun et al. (1989)[32] used back-propagation to learn the convolution kernel coefficients directly from images of hand-written numbers. Learning was thus fully automatic, performed better than manual coefficient design, and was suited to a broader range of image recognition problems and image types.
This approach became a foundation of modern computer vision.
LeNet-5, a pioneering 7-level convolutional network by LeCun et al. in 1998,[33] that classifies digits, was applied by several banks to recognize hand-written numbers on checks (British English: cheques) digitized in 32x32 pixel images. The ability to process higher resolution images requires larger and more layers of convolutional neural networks, so this technique is constrained by the availability of computing resources.
Similarly, a shift invariant neural network was proposed by W. Zhang et al. for image character recognition in 1988.[1][2] The architecture and training algorithm were modified in 1991[34] and applied for medical image processing[35] and automatic detection of breast cancer in mammograms.[36]
A different convolution-based design was proposed in 1988[37] for application to decomposition of one-dimensional electromyography convolved signals via de-convolution. This design was modified in 1989 to other de-convolution-based designs.[38][39]
The feed-forward architecture of convolutional neural networks was extended in the neural abstraction pyramid[40] by lateral and feedback connections. The resulting recurrent convolutional network allows for the flexible incorporation of contextual information to iteratively resolve local ambiguities. In contrast to previous models, image-like outputs at the highest resolution were generated, e.g., for semantic segmentation, image reconstruction, and object localization tasks.
Although CNNs were invented in the 1980s, their breakthrough in the 2000s required fast implementations on graphics processing units (GPUs).
In 2004, it was shown by K. S. Oh and K. Jung that standard neural networks can be greatly accelerated on GPUs. Their implementation was 20 times faster than an equivalent implementation on CPU.[41][23] In 2005, another paper also emphasised the value of GPGPU for machine learning.[42]
The first GPU-implementation of a CNN was described in 2006 by K. Chellapilla et al. Their implementation was 4 times faster than an equivalent implementation on CPU.[43] Subsequent work also used GPUs, initially for other types of neural networks (different from CNNs), especially unsupervised neural networks.[44][45][46][47]
In 2010, Dan Ciresan et al. at IDSIA showed that even deep standard neural networks with many layers can be quickly trained on GPU by supervised learning through the old method known as backpropagation. Their network outperformed previous machine learning methods on the MNIST handwritten digits benchmark.[48] In 2011, they extended this GPU approach to CNNs, achieving an acceleration factor of 60, with impressive results.[13] In 2011, they used such CNNs on GPU to win an image recognition contest where they achieved superhuman performance for the first time.[49] Between May 15, 2011 and September 30, 2012, their CNNs won no less than four image competitions.[50][23] In 2012, they also significantly improved on the best performance in the literature for multiple image databases, including the MNIST database, the NORB database, the HWDB1.0 dataset (Chinese characters) and the CIFAR10 dataset (dataset of 60000 32x32 labeled RGB images).[16]
Subsequently, a similar GPU-based CNN by Alex Krizhevsky et al. won the ImageNet Large Scale Visual Recognition Challenge 2012.[51] A very deep CNN with over 100 layers by Microsoft won the ImageNet 2015 contest.[52]
Compared to the training of CNNs using GPUs, not much attention was given to the Intel Xeon Phi coprocessor.[53]
A notable development is a parallelization method for training convolutional neural networks on the Intel Xeon Phi, named Controlled Hogwild with Arbitrary Order of Synchronization (CHAOS).[54]
CHAOS exploits both the thread- and SIMD-level parallelism that is available on the Intel Xeon Phi.
In the past, traditional multilayer perceptron (MLP) models have been used for image recognition.[example  needed] However, due to the full connectivity between nodes, they suffered from the curse of dimensionality, and did not scale well with higher resolution images. A 1000×1000-pixel image with RGB color channels has 3 million weights, which is too high to feasibly process efficiently at scale with full connectivity.
For example, in CIFAR-10, images are only of size 32×32×3 (32 wide, 32 high, 3 color channels), so a single fully connected neuron in a first hidden layer of a regular neural network would have 32*32*3 = 3,072 weights. A 200×200 image, however, would lead to neurons that have 200*200*3 = 120,000 weights.
Also, such network architecture does not take into account the spatial structure of data, treating input pixels which are far apart in the same way as pixels that are close together. This ignores locality of reference in image data, both computationally and semantically. Thus, full connectivity of neurons is wasteful for purposes such as image recognition that are dominated by spatially local input patterns.
Convolutional neural networks are biologically inspired variants of multilayer perceptrons that are designed to emulate the behavior of a visual cortex.[citation needed] These models mitigate the challenges posed by the MLP architecture by exploiting the strong spatially local correlation present in natural images. As opposed to MLPs, CNNs have the following distinguishing features:
Together, these properties allow CNNs to achieve better generalization on vision problems. Weight sharing dramatically reduces the number of free parameters learned, thus lowering the memory requirements for running the network and allowing the training of larger, more powerful networks.

A CNN architecture is formed by a stack of distinct layers that transform the input volume into an output volume (e.g. holding the class scores) through a differentiable function. A few distinct types of layers are commonly used. These are further discussed below.The convolutional layer is the core building block of a CNN. The layer's parameters consist of a set of learnable filters (or kernels), which have a small receptive field, but extend through the full depth of the input volume. During the forward pass, each filter is convolved across the width and height of the input volume, computing the dot product between the entries of the filter and the input and producing a 2-dimensional activation map of that filter. As a result, the network learns filters that activate when it detects some specific type of feature at some spatial position in the input.[nb 1]
Stacking the activation maps for all filters along the depth dimension forms the full output volume of the convolution layer. Every entry in the output volume can thus also be interpreted as an output of a neuron that looks at a small region in the input and shares parameters with neurons in the same activation map.
When dealing with high-dimensional inputs such as images, it is impractical to connect neurons to all neurons in the previous volume because such a network architecture does not take the spatial structure of the data into account. Convolutional networks exploit spatially local correlation by enforcing a sparse local connectivity pattern between neurons of adjacent layers: each neuron is connected to only a small region of the input volume.
The extent of this connectivity is a hyperparameter called the receptive field of the neuron. The connections are local in space (along width and height), but always extend along the entire depth of the input volume. Such an architecture ensures that the learnt filters produce the strongest response to a spatially local input pattern.
Three hyperparameters control the size of the output volume of the convolutional layer: the depth, stride and zero-padding.
The spatial size of the output volume can be computed as a function of the input volume size 



W


{\displaystyle W}

, the kernel field size of the convolutional layer neurons 



K


{\displaystyle K}

, the stride with which they are applied 



S


{\displaystyle S}

, and the amount of zero padding 



P


{\displaystyle P}

 used on the border. The formula for calculating how many neurons "fit" in a given volume is given by
If this number is not an integer, then the strides are incorrect and the neurons cannot be tiled to fit across the input volume in a symmetric way. In general, setting zero padding to be 



P
=
(
K
−
1
)

/

2


{\textstyle P=(K-1)/2}

 when the stride is 



S
=
1


{\displaystyle S=1}

 ensures that the input volume and output volume will have the same size spatially. However, it's not always completely necessary to use all of the neurons of the previous layer. For example, a neural network designer may decide to use just a portion of padding.
A parameter sharing scheme is used in convolutional layers to control the number of free parameters. It relies on the assumption that if a patch feature is useful to compute at some spatial position, then it should also be useful to compute at other positions. Denoting a single 2-dimensional slice of depth as a depth slice, the neurons in each depth slice are constrained to use the same weights and bias.
Since all neurons in a single depth slice share the same parameters, the forward pass in each depth slice of the convolutional layer can be computed as a convolution of the neuron's weights with the input volume.[nb 2] Therefore, it is common to refer to the sets of weights as a filter (or a kernel), which is convolved with the input. The result of this convolution is an activation map, and the set of activation maps for each different filter are stacked together along the depth dimension to produce the output volume. Parameter sharing contributes to the translation invariance of the CNN architecture.
Sometimes, the parameter sharing assumption may not make sense. This is especially the case when the input images to a CNN have some specific centered structure; for which we expect completely different features to be learned on different spatial locations. One practical example is when the inputs are faces that have been centered in the image: we might expect different eye-specific or hair-specific features to be learned in different parts of the image. In that case it is common to relax the parameter sharing scheme, and instead simply call the layer a "locally connected layer".
Another important concept of CNNs is pooling, which is a form of non-linear down-sampling. There are several non-linear functions to implement pooling among which max pooling is the most common. It partitions the input image into a set of non-overlapping rectangles and, for each such sub-region, outputs the maximum.
Intuitively, the exact location of a feature is less important than its rough location relative to other features. This is the idea behind the use of pooling in convolutional neural networks. The pooling layer serves to progressively reduce the spatial size of the representation, to reduce the number of parameters, memory footprint and amount of computation in the network, and hence to also control overfitting. It is common to periodically insert a pooling layer between successive convolutional layers in a CNN architecture.[citation needed] The pooling operation provides another form of translation invariance.
The pooling layer operates independently on every depth slice of the input and resizes it spatially. The most common form is a pooling layer with filters of size 2×2 applied with a stride of 2 downsamples at every depth slice in the input by 2 along both width and height, discarding 75% of the activations:
In addition to max pooling, pooling units can use other functions, such as average pooling or ℓ2-norm pooling. Average pooling was often used historically but has recently fallen out of favor compared to max pooling, which performs better in practice.[56]
Due to the aggressive reduction in the size of the representation,[which?] there is a recent trend towards using smaller filters[57] or discarding pooling layers altogether.[58]
"Region of Interest" pooling (also known as RoI pooling) is a variant of max pooling, in which output size is fixed and input rectangle is a parameter.[59]
Pooling is an important component of convolutional neural networks for object detection based on Fast R-CNN[60] architecture.
ReLU is the abbreviation of rectified linear unit, which applies the non-saturating activation function 



f
(
x
)
=
max
(
0
,
x
)


{\textstyle f(x)=\max(0,x)}

.[51] It effectively removes negative values from an activation map by setting them to zero.[61] It increases the nonlinear properties of the decision function and of the overall network without affecting the receptive fields of the convolution layer.
Other functions are also used to increase nonlinearity, for example the saturating hyperbolic tangent 



f
(
x
)
=
tanh
⁡
(
x
)


{\displaystyle f(x)=\tanh(x)}

, 



f
(
x
)
=

|

tanh
⁡
(
x
)

|



{\displaystyle f(x)=|\tanh(x)|}

, and the sigmoid function 



σ
(
x
)
=
(
1
+

e

−
x



)

−
1




{\textstyle \sigma (x)=(1+e^{-x})^{-1}}

. ReLU is often preferred to other functions because it trains the neural network several times faster without a significant penalty to generalization accuracy.[62]
Finally, after several convolutional and max pooling layers, the high-level reasoning in the neural network is done via fully connected layers. Neurons in a fully connected layer have connections to all activations in the previous layer, as seen in regular (non-convolutional) artificial neural networks. Their activations can thus be computed as an affine transformation, with matrix multiplication followed by a bias offset (vector addition of a learned or fixed bias term).
The "loss layer" specifies how training penalizes the deviation between the predicted (output) and true labels and is normally the final layer of a neural network. Various loss functions appropriate for different tasks may be used.
Softmax loss is used for predicting a single class of K mutually exclusive classes.[nb 3] Sigmoid cross-entropy loss is used for predicting K independent probability values in 



[
0
,
1
]


{\displaystyle [0,1]}

. Euclidean loss is used for regressing to real-valued labels 



(
−
∞
,
∞
)


{\displaystyle (-\infty ,\infty )}

.
CNNs use more hyperparameters than a standard multilayer perceptron (MLP). While the usual rules for learning rates and regularization constants still apply, the following should be kept in mind when optimizing.
Since feature map size decreases with depth, layers near the input layer will tend to have fewer filters while higher layers can have more. To equalize computation at each layer, the product of feature values va with pixel position is kept roughly constant across layers. Preserving more information about the input would require keeping the total number of activations (number of feature maps times number of pixel positions) non-decreasing from one layer to the next.
The number of feature maps directly controls the capacity and depends on the number of available examples and task complexity.
Common filter shapes found in the literature vary greatly, and are usually chosen based on the dataset.
The challenge is, thus, to find the right level of granularity so as to create abstractions at the proper scale, given a particular dataset, and without overfitting.
Typical values are 2×2. Very large input volumes may warrant 4×4 pooling in the lower layers.[63] However, choosing larger shapes will dramatically reduce the dimension of the signal, and may result in excess information loss. Often, non-overlapping pooling windows perform best.[56]
Regularization is a process of introducing additional information to solve an ill-posed problem or to prevent overfitting. CNNs use various types of regularization.
Because a fully connected layer occupies most of the parameters, it is prone to overfitting. One method to reduce overfitting is dropout.[64][65] At each training stage, individual nodes are either "dropped out" of the net with probability 



1
−
p


{\displaystyle 1-p}

 or kept with probability 



p


{\displaystyle p}

, so that a reduced network is left; incoming and outgoing edges to a dropped-out node are also removed. Only the reduced network is trained on the data in that stage. The removed nodes are then reinserted into the network with their original weights.
In the training stages, the probability that a hidden node will be dropped is usually 0.5; for input nodes, this should be much lower, intuitively because information is directly lost when input nodes are ignored.
At testing time after training has finished, we would ideally like to find a sample average of all possible 




2

n




{\displaystyle 2^{n}}

 dropped-out networks; unfortunately this is unfeasible for large values of 



n


{\displaystyle n}

. However, we can find an approximation by using the full network with each node's output weighted by a factor of 



p


{\displaystyle p}

, so the expected value of the output of any node is the same as in the training stages. This is the biggest contribution of the dropout method: although it effectively generates 




2

n




{\displaystyle 2^{n}}

 neural nets, and as such allows for model combination, at test time only a single network needs to be tested.
By avoiding training all nodes on all training data, dropout decreases overfitting. The method also significantly improves training speed. This makes the model combination practical, even for deep neural networks. The technique seems to reduce node interactions, leading them to learn more robust features[clarification needed] that better generalize to new data.
DropConnect is the generalization of dropout in which each connection, rather than each output unit, can be dropped with probability 



1
−
p


{\displaystyle 1-p}

. Each unit thus receives input from a random subset of units in the previous layer.[66]
DropConnect is similar to dropout as it introduces dynamic sparsity within the model, but differs in that the sparsity is on the weights, rather than the output vectors of a layer. In other words, the fully connected layer with DropConnect becomes a sparsely connected layer in which the connections are chosen at random during the training stage.
A major drawback to Dropout is that it does not have the same benefits for convolutional layers, where the neurons are not fully connected.
In stochastic pooling,[67] the conventional deterministic pooling operations are replaced with a stochastic procedure, where the activation within each pooling region is picked randomly according to a multinomial distribution, given by the activities within the pooling region. This approach is free of hyperparameters and can be combined with other regularization approaches, such as dropout and data augmentation.
An alternate view of stochastic pooling is that it is equivalent to standard max pooling but with many copies of an input image, each having small local deformations. This is similar to explicit elastic deformations of the input images,[68] which delivers excellent performance on the MNIST data set.[68] Using stochastic pooling in a multilayer model gives an exponential number of deformations since the selections in higher layers are independent of those below.
Since the degree of model overfitting is determined by both its power and the amount of training it receives, providing a convolutional network with more training examples can reduce overfitting. Since these networks are usually trained with all available data, one approach is to either generate new data from scratch (if possible) or perturb existing data to create new ones. For example, input images could be asymmetrically cropped by a few percent to create new examples with the same label as the original.[69]
One of the simplest methods to prevent overfitting of a network is to simply stop the training before overfitting has had a chance to occur. It comes with the disadvantage that the learning process is halted.
Another simple way to prevent overfitting is to limit the number of parameters, typically by limiting the number of hidden units in each layer or limiting network depth. For convolutional networks, the filter size also affects the number of parameters. Limiting the number of parameters restricts the predictive power of the network directly, reducing the complexity of the function that it can perform on the data, and thus limits the amount of overfitting. This is equivalent to a "zero norm".
A simple form of added regularizer is weight decay, which simply adds an additional error, proportional to the sum of weights (L1 norm) or squared magnitude (L2 norm) of the weight vector, to the error at each node. The level of acceptable model complexity can be reduced by increasing the proportionality constant, thus increasing the penalty for large weight vectors.
L2 regularization is the most common form of regularization. It can be implemented by penalizing the squared magnitude of all parameters directly in the objective. The L2 regularization has the intuitive interpretation of heavily penalizing peaky weight vectors and preferring diffuse weight vectors. Due to multiplicative interactions between weights and inputs this has the useful property of encouraging the network to use all of its inputs a little rather than some of its inputs a lot.
L1 regularization is another common form. It is possible to combine L1 with L2 regularization (this is called Elastic net regularization). The L1 regularization leads the weight vectors to become sparse during optimization. In other words, neurons with L1 regularization end up using only a sparse subset of their most important inputs and become nearly invariant to the noisy inputs.
Another form of regularization is to enforce an absolute upper bound on the magnitude of the weight vector for every neuron and use projected gradient descent to enforce the constraint. In practice, this corresponds to performing the parameter update as normal, and then enforcing the constraint by clamping the weight vector 






w
→





{\displaystyle {\vec {w}}}

 of every neuron to satisfy 



‖



w
→




‖

2


<
c


{\displaystyle \|{\vec {w}}\|_{2}<c}

. Typical values of 



c


{\displaystyle c}

 are order of 3–4. Some papers report improvements[70] when using this form of regularization.
Pooling loses the precise spatial relationships between high-level parts (such as nose and mouth in a face image). These relationships are needed for identity recognition. Overlapping the pools so that each feature occurs in multiple pools, helps retain the information. Translation alone cannot extrapolate the understanding of geometric relationships to a radically new viewpoint, such as a different orientation or scale. On the other hand, people are very good at extrapolating; after seeing a new shape once they can recognize it from a different viewpoint.[71]
Currently, the common way to deal with this problem is to train the network on transformed data in different orientations, scales, lighting, etc. so that the network can cope with these variations. This is computationally intensive for large data-sets. The alternative is to use a hierarchy of coordinate frames and to use a group of neurons to represent a conjunction of the shape of the feature and its pose relative to the retina. The pose relative to retina is the relationship between the coordinate frame of the retina and the intrinsic features' coordinate frame.[72]
Thus, one way of representing something is to embed the coordinate frame within it. Once this is done, large features can be recognized by using the consistency of the poses of their parts (e.g. nose and mouth poses make a consistent prediction of the pose of the whole face). Using this approach ensures that the higher level entity (e.g. face) is present when the lower level (e.g. nose and mouth) agree on its prediction of the pose. The vectors of neuronal activity that represent pose ("pose vectors") allow spatial transformations modeled as linear operations that make it easier for the network to learn the hierarchy of visual entities and generalize across viewpoints. This is similar to the way the human visual system imposes coordinate frames in order to represent shapes.[73]
CNNs are often used in image recognition systems. In 2012 an error rate of 0.23 percent on the MNIST database was reported.[16] Another paper on using CNN for image classification reported that the learning process was "surprisingly fast"; in the same paper, the best published results as of 2011 were achieved in the MNIST database and the NORB database.[13] Subsequently, a similar CNN called 
AlexNet[74] won the ImageNet Large Scale Visual Recognition Challenge 2012.
When applied to facial recognition, CNNs achieved a large decrease in error rate.[75] Another paper reported a 97.6 percent recognition rate on "5,600 still images of more than 10 subjects".[9] CNNs were used to assess video quality in an objective way after manual training; the resulting system had a very low root mean square error.[29]
The ImageNet Large Scale Visual Recognition Challenge is a benchmark in object classification and detection, with millions of images and hundreds of object classes. In the ILSVRC 2014,[76] a large-scale visual recognition challenge, almost every highly ranked team used CNN as their basic framework. The winner GoogLeNet[77] (the foundation of DeepDream) increased the mean average precision of object detection to 0.439329, and reduced classification error to 0.06656, the best result to date. Its network applied more than 30 layers. That performance of convolutional neural networks on the ImageNet tests was close to that of humans.[78] The best algorithms still struggle with objects that are small or thin, such as a small ant on a stem of a flower or a person holding a quill in their hand. They also have trouble with images that have been distorted with filters, an increasingly common phenomenon with modern digital cameras. By contrast, those kinds of images rarely trouble humans. Humans, however, tend to have trouble with other issues. For example, they are not good at classifying objects into fine-grained categories such as the particular breed of dog or species of bird, whereas convolutional neural networks handle this.[citation needed]
In 2015 a many-layered CNN demonstrated the ability to spot faces from a wide range of angles, including upside down, even when partially occluded, with competitive performance. The network was trained on a database of 200,000 images that included faces at various angles and orientations and a further 20 million images without faces. They used batches of 128 images over 50,000 iterations.[79]
Compared to image data domains, there is relatively little work on applying CNNs to video classification. Video is more complex than images since it has another (temporal) dimension. However, some extensions of CNNs into the video domain have been explored. One approach is to treat space and time as equivalent dimensions of the input and perform convolutions in both time and space.[80][81] Another way is to fuse the features of two convolutional neural networks, one for the spatial and one for the temporal stream.[82][83][84] Long short-term memory (LSTM) recurrent units are typically incorporated after the CNN to account for inter-frame or inter-clip dependencies.[85][86] Unsupervised learning schemes for training spatio-temporal features have been introduced, based on Convolutional Gated Restricted Boltzmann Machines[87] and Independent Subspace Analysis.[88]
CNNs have also been explored for natural language processing. CNN models are effective for various NLP problems and achieved excellent results in semantic parsing,[89] search query retrieval,[90] sentence modeling,[91] classification,[92] prediction[93] and other traditional NLP tasks.[94]
A CNN with 1-D convolutions was used on time series in the frequency domain (spectral residual) by an unsupervised model to detect anomalies in the time domain.[95]
CNNs have been used in drug discovery. Predicting the interaction between molecules and biological proteins can identify potential treatments. In 2015, Atomwise introduced AtomNet, the first deep learning neural network for structure-based rational drug design.[96] The system trains directly on 3-dimensional representations of chemical interactions. Similar to how image recognition networks learn to compose smaller, spatially proximate features into larger, complex structures,[97] AtomNet discovers chemical features, such as aromaticity, sp3 carbons and hydrogen bonding. Subsequently, AtomNet was used to predict novel candidate biomolecules for multiple disease targets, most notably treatments for the Ebola virus[98] and multiple sclerosis.[99]
CNNs can be naturally tailored to analyze a sufficiently large collection of time series data representing one-week-long human physical activity streams augmented by the rich clinical data (including the death register, as provided by, e.g., the NHANES study). A simple CNN was combined with Cox-Gompertz proportional hazards model and used to produce a proof-of-concept example of digital biomarkers of aging in the form of all-causes-mortality predictor.[100]
CNNs have been used in the game of checkers. From 1999 to 2001, Fogel and Chellapilla published papers showing how a convolutional neural network could learn to play checker using co-evolution. The learning process did not use prior human professional games, but rather focused on a minimal set of information contained in the checkerboard: the location and type of pieces, and the difference in number of pieces between the two sides. Ultimately, the program (Blondie24) was tested on 165 games against players and ranked in the highest 0.4%.[101][102] It also earned a win against the program Chinook at its "expert" level of play.[103]
CNNs have been used in computer Go. In December 2014, Clark and Storkey published a paper showing that a CNN trained by supervised learning from a database of human professional games could outperform GNU Go and win some games against Monte Carlo tree search Fuego 1.1 in a fraction of the time it took Fuego to play.[104] Later it was announced that a large 12-layer convolutional neural network had correctly predicted the professional move in 55% of positions, equalling the accuracy of a 6 dan human player. When the trained convolutional network was used directly to play games of Go, without any search, it beat the traditional search program GNU Go in 97% of games, and matched the performance of the Monte Carlo tree search program Fuego simulating ten thousand playouts (about a million positions) per move.[105]
A couple of CNNs for choosing moves to try ("policy network") and evaluating positions ("value network") driving MCTS were used by AlphaGo, the first to beat the best human player at the time.[106]
Recurrent neural networks are generally considered the best neural network architectures for time series forecasting (and sequence modeling in general), but recent studies show that convolutional networks can perform comparably or even better.[107][5] Dilated convolutions[108] might enable one-dimensional convolutional neural networks to effectively learn time series dependences.[109] Convolutions can be implemented more efficiently than RNN-based solutions, and they do not suffer from vanishing (or exploding) gradients.[110] Convolutional networks can provide an improved forecasting performance when there are multiple similar time series to learn from.[111] CNNs can also be applied to further tasks in time series analysis (e.g., time series classification[112] or quantile forecasting[113]).
For many applications, the training data is less available. Convolutional neural networks usually require a large amount of training data in order to avoid overfitting. A common technique is to train the network on a larger data set from a related domain. Once the network parameters have converged an additional training step is performed using the in-domain data to fine-tune the network weights. This allows convolutional networks to be successfully applied to problems with small training sets.[114]
End-to-end training and prediction are common practice in computer vision. However, human interpretable explanations are required for critical systems such as a self-driving cars.[115] With recent advances in visual salience, spatial and temporal attention, the most critical spatial regions/temporal instants could be visualized to justify the CNN predictions.[116][117]
A deep Q-network (DQN) is a type of deep learning model that combines a deep neural network with Q-learning, a form of reinforcement learning. Unlike earlier reinforcement learning agents, DQNs that utilize CNNs can learn directly from high-dimensional sensory inputs.[citation needed]
Preliminary results were presented in 2014, with an accompanying paper in February 2015.[118] The research described an application to Atari 2600 gaming. Other deep reinforcement learning models preceded it.[119]
Convolutional deep belief networks (CDBN) have structure very similar to convolutional neural networks and are trained similarly to deep belief networks. Therefore, they exploit the 2D structure of images, like CNNs do, and make use of pre-training like deep belief networks. They provide a generic structure that can be used in many image and signal processing tasks. Benchmark results on standard image datasets like CIFAR[120] have been obtained using CDBNs.[121]
qX�W  A recurrent neural network (RNN) is a class of artificial neural networks where connections between nodes form a directed graph along a temporal sequence. This allows it to exhibit temporal dynamic behavior. Derived from feedforward neural networks, RNNs can use their internal state (memory) to process variable length sequences of inputs[1]. This makes them applicable to tasks such as unsegmented, connected handwriting recognition[2] or speech recognition.[3][4]
The term “recurrent neural network” is used indiscriminately to refer to two broad classes of networks with a similar general structure, where one is finite impulse and the other is infinite impulse. Both classes of networks exhibit temporal dynamic behavior.[5] A finite impulse recurrent network is a directed acyclic graph that can be unrolled and replaced with a strictly feedforward neural network, while an infinite impulse recurrent network is a directed cyclic graph that can not be unrolled.
Both finite impulse and infinite impulse recurrent networks can have additional stored states, and the storage can be under direct control by the neural network. The storage can also be replaced by another network or graph, if that incorporates time delays or has feedback loops. Such controlled states are referred to as gated state or gated memory, and are part of long short-term memory networks (LSTMs) and gated recurrent units. This is also called Feedback Neural Network. 
Recurrent neural networks were based on David Rumelhart's work in 1986.[6] Hopfield networks - a special kind of RNN - were discovered by John Hopfield in 1982. In 1993, a neural history compressor system solved a “Very Deep Learning” task that required more than 1000 subsequent layers in an RNN unfolded in time.[7]
Long short-term memory (LSTM) networks were discovered by Hochreiter and Schmidhuber in 1997 and set accuracy records in multiple applications domains.[8]
Around 2007, LSTM started to revolutionize speech recognition, outperforming traditional models in certain speech applications.[9] In 2009, a Connectionist Temporal Classification (CTC)-trained LSTM network was the first RNN to win pattern recognition contests when it won several competitions in connected handwriting recognition.[10][11] In 2014, the Chinese search giant Baidu used CTC-trained RNNs to break the Switchboard Hub5'00 speech recognition benchmark without using any traditional speech processing methods.[12]
LSTM also improved large-vocabulary speech recognition[3][4] and text-to-speech synthesis[13] and was used in Google Android.[10][14] In 2015, Google's speech recognition reportedly experienced a dramatic performance jump of 49%[citation needed] through CTC-trained LSTM, which was used by Google voice search.[15]
LSTM broke records for improved machine translation,[16] Language Modeling[17] and Multilingual Language Processing.[18] LSTM combined with convolutional neural networks (CNNs) improved automatic image captioning.[19]
RNNs come in many variants.
Basic RNNs are a network of neuron-like nodes organized into successive layers. Each node in a given layer is connected with a directed (one-way) connection to every other node in the next successive layer.[citation needed] Each node (neuron) has a time-varying real-valued activation. Each connection (synapse) has a modifiable real-valued weight. Nodes are either input nodes (receiving data from outside the network), output nodes (yielding results), or hidden nodes (that modify the data en route from input to output).
For supervised learning in discrete time settings, sequences of real-valued input vectors arrive at the input nodes, one vector at a time. At any given time step, each non-input unit computes its current activation (result) as a nonlinear function of the weighted sum of the activations of all units that connect to it. Supervisor-given target activations can be supplied for some output units at certain time steps. For example, if the input sequence is a speech signal corresponding to a spoken digit, the final target output at the end of the sequence may be a label classifying the digit.
In reinforcement learning settings, no teacher provides target signals. Instead a fitness function or reward function is occasionally used to evaluate the RNN's performance, which influences its input stream through output units connected to actuators that affect the environment. This might be used to play a game in which progress is measured with the number of points won.
Each sequence produces an error as the sum of the deviations of all target signals from the corresponding activations computed by the network. For a training set of numerous sequences, the total error is the sum of the errors of all individual sequences.
An Elman network is a three-layer network (arranged horizontally as x, y, and z in the illustration) with the addition of a set of context units (u in the illustration). The middle (hidden) layer is connected to these context units fixed with a weight of one.[20] At each time step, the input is fed forward and a learning rule is applied. The fixed back-connections save a copy of the previous values of the hidden units in the context units (since they propagate over the connections before the learning rule is applied). Thus the network can maintain a sort of state, allowing it to perform such tasks as sequence-prediction that are beyond the power of a standard multilayer perceptron.
Jordan networks are similar to Elman networks. The context units are fed from the output layer instead of the hidden layer. The context units in a Jordan network are also referred to as the state layer. They have a recurrent connection to themselves.[20]
Elman and Jordan networks are also known as “simple recurrent networks” (SRN).
Variables and functions
The Hopfield network is an RNN in which all connections are symmetric. It requires stationary inputs and is thus not a general RNN, as it does not process sequences of patterns. It guarantees that it will converge. If the connections are trained using Hebbian learning then the Hopfield network can perform as robust content-addressable memory, resistant to connection alteration.
Introduced by Bart Kosko,[23] a bidirectional associative memory (BAM) network is a variant of a Hopfield network that stores associative data as a vector. The bi-directionality comes from passing information through a matrix and its transpose. Typically, bipolar encoding is preferred to binary encoding of the associative pairs. Recently, stochastic BAM models using Markov stepping were optimized for increased network stability and relevance to real-world applications.[24]
A BAM network has two layers, either of which can be driven as an input to recall an association and produce an output on the other layer.[25]
The echo state network (ESN) has a sparsely connected random hidden layer. The weights of output neurons are the only part of the network that can change (be trained). ESNs are good at reproducing certain time series.[26] A variant for spiking neurons is known as a liquid state machine.[27]
The Independently recurrent neural network (IndRNN)[28] addresses the gradient vanishing and exploding problems in the traditional fully connected RNN. Each neuron in one layer only receives its own past state as context information (instead of full connectivity to all other neurons in this layer) and thus neurons are independent of each other's history. The gradient backpropagation can be regulated to avoid gradient vanishing and exploding in order to keep long or short-term memory. The cross-neuron information is explored in the next layers. IndRNN can be robustly trained with the non-saturated nonlinear functions such as ReLU. Using skip connections, deep networks can be trained.
A recursive neural network[29] is created by applying the same set of weights recursively over a differentiable graph-like structure by traversing the structure in topological order. Such networks are typically also trained by the reverse mode of automatic differentiation.[30][31] They can process distributed representations of structure, such as logical terms. A special case of recursive neural networks is the RNN whose structure corresponds to a linear chain. Recursive neural networks have been applied to natural language processing.[32] The Recursive Neural Tensor Network uses a tensor-based composition function for all nodes in the tree.[33]
The neural history compressor is an unsupervised stack of RNNs.[34] At the input level, it learns to predict its next input from the previous inputs. Only unpredictable inputs of some RNN in the hierarchy become inputs to the next higher level RNN, which therefore recomputes its internal state only rarely. Each higher level RNN thus studies a compressed representation of the information in the RNN below. This is done such that the input sequence can be precisely reconstructed from the representation at the highest level.
The system effectively minimises the description length or the negative logarithm of the probability of the data.[35] Given a lot of learnable predictability in the incoming data sequence, the highest level RNN can use supervised learning to easily classify even deep sequences with long intervals between important events.
It is possible to distill the RNN hierarchy into two RNNs: the "conscious" chunker (higher level) and the "subconscious" automatizer (lower level).[34] Once the chunker has learned to predict and compress inputs that are unpredictable by the automatizer, then the automatizer can be forced in the next learning phase to predict or imitate through additional units the hidden units of the more slowly changing chunker. This makes it easy for the automatizer to learn appropriate, rarely changing memories across long intervals. In turn this helps the automatizer to make many of its once unpredictable inputs predictable, such that the chunker can focus on the remaining unpredictable events.[34]
A generative model partially overcame the vanishing gradient problem[36] of automatic differentiation or backpropagation in neural networks in 1992. In 1993, such a system solved a “Very Deep Learning” task that required more than 1000 subsequent layers in an RNN unfolded in time.[7]
Second order RNNs use higher order weights 



w




i
j
k




{\displaystyle w{}_{ijk}}

 instead of the standard 



w




i
j




{\displaystyle w{}_{ij}}

 weights, and states can be a product. This allows a direct mapping to a finite state machine both in training, stability, and representation.[37][38] Long short-term memory is an example of this but has no such formal mappings or proof of stability.
Long short-term memory (LSTM) is a deep learning system that avoids the vanishing gradient problem. LSTM is normally augmented by recurrent gates called “forget gates”.[39] LSTM prevents backpropagated errors from vanishing or exploding.[36] Instead, errors can flow backwards through unlimited numbers of virtual layers unfolded in space. That is, LSTM can learn tasks[10] that require memories of events that happened thousands or even millions of discrete time steps earlier. Problem-specific LSTM-like topologies can be evolved.[40] LSTM works even given long delays between significant events and can handle signals that mix low and high frequency components.
Many applications use stacks of LSTM RNNs[41] and train them by Connectionist Temporal Classification (CTC)[42] to find an RNN weight matrix that maximizes the probability of the label sequences in a training set, given the corresponding input sequences. CTC achieves both alignment and recognition.
LSTM can learn to recognize context-sensitive languages unlike previous models based on hidden Markov models (HMM) and similar concepts.[43]
Gated recurrent units (GRUs) are a gating mechanism in recurrent neural networks introduced in 2014. They are used in the full form and several simplified variants.[44][45] Their performance on polyphonic music modeling and speech signal modeling was found to be similar to that of long short-term memory.[46] They have fewer parameters than LSTM, as they lack an output gate.[47]
Bi-directional RNNs use a finite sequence to predict or label each element of the sequence based on the element's past and future contexts. This is done by concatenating the outputs of two RNNs, one processing the sequence from left to right, the other one from right to left. The combined outputs are the predictions of the teacher-given target signals. This technique proved to be especially useful when combined with LSTM RNNs.[48][49]
A continuous time recurrent neural network (CTRNN) uses a system of ordinary differential equations to model the effects on a neuron of the incoming spike train.
For a neuron 



i


{\displaystyle i}

 in the network with action potential 




y

i




{\displaystyle y_{i}}

, the rate of change of activation is given by:
Where:
CTRNNs have been applied to evolutionary robotics where they have been used to address vision,[50] co-operation,[51] and minimal cognitive behaviour.[52]
Note that, by the Shannon sampling theorem, discrete time recurrent neural networks can be viewed as continuous-time recurrent neural networks where the differential equations have transformed into equivalent difference equations.[53] This transformation can be thought of as occurring after the post-synaptic node activation functions 




y

i


(
t
)


{\displaystyle y_{i}(t)}

 have been low-pass filtered but prior to sampling.
Hierarchical RNNs connect their neurons in various ways to decompose hierarchical behavior into useful subprograms.[34][54]
Generally, a Recurrent Multi-Layer Perceptron (RMLP) network consists of cascaded subnetworks, each of which contains multiple layers of nodes. Each of these subnetworks is feed-forward except for the last layer, which can have feedback connections. Each of these subnets is connected only by feed forward connections.[55]
A multiple timescales recurrent neural network (MTRNN) is a neural-based computational model that can simulate the functional hierarchy of the brain through self-organization that depends on spatial connection between neurons and on distinct types of neuron activities, each with distinct time properties.[56][57] With such varied neuronal activities, continuous sequences of any set of behaviors are segmented into reusable primitives, which in turn are flexibly integrated into diverse sequential behaviors. The biological approval of such a type of hierarchy was discussed in the memory-prediction theory of brain function by Hawkins in his book On Intelligence.[citation needed]
Neural Turing machines (NTMs) are a method of extending recurrent neural networks by coupling them to external memory resources which they can interact with by attentional processes. The combined system is analogous to a Turing machine or Von Neumann architecture but is differentiable end-to-end, allowing it to be efficiently trained with gradient descent.[58]
Differentiable neural computers (DNCs) are an extension of Neural Turing machines, allowing for usage of fuzzy amounts of each memory address and a record of chronology.
Neural network pushdown automata (NNPDA) are similar to NTMs, but tapes are replaced by analogue stacks that are differentiable and that are trained. In this way, they are similar in complexity to recognizers of context free grammars (CFGs).[59]
Greg Snider of HP Labs describes a system of cortical computing with memristive nanodevices.[60] The memristors (memory resistors) are implemented by thin film materials in which the resistance is electrically tuned via the transport of ions or oxygen vacancies within the film. DARPA's SyNAPSE project has funded IBM Research and HP Labs, in collaboration with the Boston University Department of Cognitive and Neural Systems (CNS), to develop neuromorphic architectures which may be based on memristive systems.
Memristive networks are a particular type of physical neural network that have very similar properties to (Little-)Hopfield networks, as they have a continuous dynamics, have a limited memory capacity and they natural relax via the minimization of a function which is asymptotic to the Ising model. In this sense, the dynamics of a memristive circuit has the advantage compared to a Resistor-Capacitor network to have a more interesting non-linear behavior. From this point of view, engineering an analog memristive networks accounts to a peculiar type of neuromorphic engineering in which the device behavior depends on the circuit wiring, or topology.
[61][62]
Gradient descent is a first-order iterative optimization algorithm for finding the minimum of a function. In neural networks, it can be used to minimize the error term by changing each weight in proportion to the derivative of the error with respect to that weight, provided the non-linear activation functions are differentiable. Various methods for doing so were developed in the 1980s and early 1990s by Werbos, Williams, Robinson, Schmidhuber, Hochreiter, Pearlmutter and others.
The standard method is called “backpropagation through time” or BPTT, and is a generalization of back-propagation for feed-forward networks.[63][64] Like that method, it is an instance of automatic differentiation in the reverse accumulation mode of Pontryagin's minimum principle. A more computationally expensive online variant is called “Real-Time Recurrent Learning” or RTRL,[65][66] which is an instance of automatic differentiation in the forward accumulation mode with stacked tangent vectors. Unlike BPTT, this algorithm is local in time but not local in space.
In this context, local in space means that a unit's weight vector can be updated using only information stored in the connected units and the unit itself such that update complexity of a single unit is linear in the dimensionality of the weight vector. Local in time means that the updates take place continually (on-line) and depend only on the most recent time step rather than on multiple time steps within a given time horizon as in BPTT. Biological neural networks appear to be local with respect to both time and space.[67][68]
For recursively computing the partial derivatives, RTRL has a time-complexity of O(number of hidden x number of weights) per time step for computing the Jacobian matrices, while BPTT only takes O(number of weights) per time step, at the cost of storing all forward activations within the given time horizon.[69] An online hybrid between BPTT and RTRL with intermediate complexity exists,[70][71] along with variants for continuous time.[72]
A major problem with gradient descent for standard RNN architectures is that error gradients vanish exponentially quickly with the size of the time lag between important events.[36][73] LSTM combined with a BPTT/RTRL hybrid learning method attempts to overcome these problems.[8] This problem is also solved in the independently recurrent neural network (IndRNN)[28] by reducing the context of a neuron to its own past state and the cross-neuron information can then be explored in the following layers. Memories of different range including long-term memory can be learned without the gradient vanishing and exploding problem.
The on-line algorithm called causal recursive backpropagation (CRBP), implements and combines BPTT and RTRL paradigms for locally recurrent networks.[74] It works with the most general locally recurrent networks. The CRBP algorithm can minimize the global error term. This fact improves stability of the algorithm, providing a unifying view on gradient calculation techniques for recurrent networks with local feedback.
One approach to the computation of gradient information in RNNs with arbitrary architectures is based on signal-flow graphs diagrammatic derivation.[75] It uses the BPTT batch algorithm, based on Lee's theorem for network sensitivity calculations.[76] It was proposed by Wan and Beaufays, while its fast online version was proposed by Campolucci, Uncini and Piazza.[76]
Training the weights in a neural network can be modeled as a non-linear global optimization problem. A target function can be formed to evaluate the fitness or error of a particular weight vector as follows: First, the weights in the network are set according to the weight vector. Next, the network is evaluated against the training sequence. Typically, the sum-squared-difference between the predictions and the target values specified in the training sequence is used to represent the error of the current weight vector. Arbitrary global optimization techniques may then be used to minimize this target function.
The most common global optimization method for training RNNs is genetic algorithms, especially in unstructured networks.[77][78][79]
Initially, the genetic algorithm is encoded with the neural network weights in a predefined manner where one gene in the chromosome represents one weight link. The whole network is represented as a single chromosome. The fitness function is evaluated as follows:
Many chromosomes make up the population; therefore, many different neural networks are evolved until a stopping criterion is satisfied. A common stopping scheme is: 
The stopping criterion is evaluated by the fitness function as it gets the reciprocal of the mean-squared-error from each network during training. Therefore, the goal of the genetic algorithm is to maximize the fitness function, reducing the mean-squared-error.
Other global (and/or evolutionary) optimization techniques may be used to seek a good set of weights, such as simulated annealing or particle swarm optimization.
RNNs may behave chaotically. In such cases, dynamical systems theory may be used for analysis.
They are in fact recursive neural networks with a particular structure: that of a linear chain. Whereas recursive neural networks operate on any hierarchical structure, combining child representations into parent representations, recurrent neural networks operate on the linear progression of time, combining the previous time step and a hidden representation into the representation for the current time step.
In particular, RNNs can appear as nonlinear versions of finite impulse response and infinite impulse response filters and also as a nonlinear autoregressive exogenous model (NARX).[80]
Applications of Recurrent Neural Networks include:
qX1&  Long short-term memory (LSTM) is an artificial recurrent neural network (RNN) architecture[1] used in the field of deep learning. Unlike standard feedforward neural networks, LSTM has feedback connections. It can not only process single data points (such as images), but also entire sequences of data (such as speech or video). For example, LSTM is applicable to tasks such as unsegmented, connected handwriting recognition[2], speech recognition[3][4] and anomaly detection in network traffic or IDS's (intrusion detection systems).
A common LSTM unit is composed of a cell, an input gate, an output gate and a forget gate. The cell remembers values over arbitrary time intervals and the three gates regulate the flow of information into and out of the cell.
LSTM networks are well-suited to classifying, processing and making predictions based on time series data, since there can be lags of unknown duration between important events in a time series. LSTMs were developed to deal with the exploding and vanishing gradient problems that can be encountered when training traditional RNNs. Relative insensitivity to gap length is an advantage of LSTM over RNNs, hidden Markov models and other sequence learning methods in numerous applications.[citation needed]
LSTM was proposed in 1997 by Sepp Hochreiter and Jürgen Schmidhuber.[1] By introducing Constant Error Carousel (CEC) units, LSTM deals with the exploding and vanishing gradient problems. The initial version of LSTM block included cells, input and output gates.[5]
In 1999, Felix Gers and his advisor Jürgen Schmidhuber and Fred Cummins introduced the forget gate (also called “keep gate”) into LSTM architecture,[6] 
enabling the LSTM to reset its own state.[5]
In 2000, Gers & Schmidhuber & Cummins added peephole connections (connections from the cell to the gates) into the architecture.[7] Additionally, the output activation function was omitted.[5]
In 2014, Kyunghyun Cho et al. put forward a simplified variant called Gated recurrent unit (GRU).[8]
Among other successes, LSTM achieved record results in natural language text compression,[9] unsegmented connected handwriting recognition[10] and won the ICDAR handwriting competition (2009). LSTM networks were a major component of a network that achieved a record 17.7% phoneme error rate on the classic TIMIT natural speech dataset (2013).[11]
As of 2016, major technology companies including Google, Apple, and Microsoft were using LSTMs as fundamental components in new products.[12] For example, Google used LSTM for speech recognition on the smartphone,[13][14] for the smart assistant Allo[15] and for Google Translate.[16][17] Apple uses LSTM for the "Quicktype" function on the iPhone[18][19] and for Siri.[20] Amazon uses LSTM for Amazon Alexa.[21]
In 2017, Facebook performed some 4.5 billion automatic translations every day using long short-term memory networks.[22]
In 2017, researchers from Michigan State University, IBM Research, and Cornell University published a study in the Knowledge Discovery and Data Mining (KDD) conference.[23][24][25] Their study describes a novel neural network that performs better on certain data sets than the widely used long short-term memory neural network.
Further in 2017 Microsoft reported reaching 95.1% recognition accuracy on the Switchboard corpus, incorporating a vocabulary of 165,000 words. The approach used "dialog session-based long-short-term memory".[26]
In 2019, researchers from the University of Waterloo proposed a related RNN architecture, derived using the Legendre polynomials, that represents continuous windows of time, and outperforms the LSTM on some memory-related benchmarks.[27]
In theory, classic (or "vanilla") RNNs can keep track of arbitrary long-term dependencies in the input sequences. The problem of vanilla RNNs is computational (or practical) in nature: when training a vanilla RNN using back-propagation, the gradients which are back-propagated can "vanish" (that is, they can tend to zero) or "explode" (that is, they can tend to infinity), because of the computations involved in the process, which use finite-precision numbers. RNNs using LSTM units partially solve the vanishing gradient problem, because LSTM units allow gradients to also flow unchanged. However, LSTM networks can still suffer from the exploding gradient problem.[28]
There are several architectures of LSTM units. A common architecture is composed of a cell (the memory part of the LSTM unit) and three "regulators", usually called gates, of the flow of information inside the LSTM unit: an input gate, an output gate and a forget gate. Some variations of the LSTM unit do not have one or more of these gates or maybe have other gates. For example, gated recurrent units (GRUs) do not have an output gate.
Intuitively, the cell is responsible for keeping track of the dependencies between the elements in the input sequence. The input gate controls the extent to which a new value flows into the cell, the forget gate controls the extent to which a value remains in the cell and the output gate controls the extent to which the value in the cell is used to compute the output activation of the LSTM unit. The activation function of the LSTM gates is often the logistic sigmoid function.
There are connections into and out of the LSTM gates, a few of which are recurrent. The weights of these connections, which need to be learned during training, determine how the gates operate.
In the equations below, the lowercase variables represent vectors. Matrices 




W

q




{\displaystyle W_{q}}

 and 




U

q




{\displaystyle U_{q}}

 contain, respectively, the weights of the input and recurrent connections, where the subscript 






q




{\displaystyle _{q}}

 can either be the input gate 



i


{\displaystyle i}

, output gate 



o


{\displaystyle o}

, the forget gate 



f


{\displaystyle f}

 or the memory cell 



c


{\displaystyle c}

, depending on the activation being calculated. In this section, we are thus using a "vector notation". So, for example, 




c

t


∈


R


h




{\displaystyle c_{t}\in \mathbb {R} ^{h}}

 is not just one cell of one LSTM unit, but contains 



h


{\displaystyle h}

 LSTM unit's cells.
The compact forms of the equations for the forward pass of an LSTM unit with a forget gate are:[1][7]
where the initial values are 




c

0


=
0


{\displaystyle c_{0}=0}

 and 




h

0


=
0


{\displaystyle h_{0}=0}

 and the operator 



∘


{\displaystyle \circ }

 denotes the Hadamard product (element-wise product). The subscript 



t


{\displaystyle t}

 indexes the time step.
where the superscripts 



d


{\displaystyle d}

 and 



h


{\displaystyle h}

 refer to the number of input features and number of hidden units, respectively.
The figure on the right is a graphical representation of an LSTM unit with peephole connections (i.e. a peephole LSTM).[29][30] Peephole connections allow the gates to access the constant error carousel (CEC), whose activation is the cell state.[31] 




h

t
−
1




{\displaystyle h_{t-1}}

 is not used, 




c

t
−
1




{\displaystyle c_{t-1}}

 is used instead in most places.
Peephole convolutional LSTM.[32] The 



∗


{\displaystyle *}

 denotes the convolution operator.
An RNN using LSTM units can be trained in a supervised fashion, on a set of training sequences, using an optimization algorithm, like gradient descent, combined with backpropagation through time to compute the gradients needed during the optimization process, in order to change each weight of the LSTM network in proportion to the derivative of the error (at the output layer of the LSTM network) with respect to corresponding weight.
A problem with using gradient descent for standard RNNs is that error gradients vanish exponentially quickly with the size of the time lag between important events. This is due to 




lim

n
→
∞



W

n


=
0


{\displaystyle \lim _{n\to \infty }W^{n}=0}

 if the spectral radius of 



W


{\displaystyle W}

 is smaller than 1.[33][34]
However, with LSTM units, when error values are back-propagated from the output layer, the error remains in the LSTM unit's cell. This "error carousel" continuously feeds error back to each of the LSTM unit's gates, until they learn to cut off the value.
Many applications use stacks of LSTM RNNs[35] and train them by connectionist temporal classification (CTC)[36] to find an RNN weight matrix that maximizes the probability of the label sequences in a training set, given the corresponding input sequences. CTC achieves both alignment and recognition.
Sometimes, it can be advantageous to train (parts of) an LSTM by neuroevolution[37] or by policy gradient methods, especially when there is no "teacher" (that is, training labels).
There have been several successful stories of training, in a non-supervised fashion, RNNs with LSTM units.
In 2018, Bill Gates called it a “huge milestone in advancing artificial intelligence” when bots developed by OpenAI were able to beat humans in the game of Dota 2.[38] OpenAI Five consists of five independent but coordinated neural networks. Each network is trained by a policy gradient method without supervising teacher and contains a single-layer, 1024-unit Long-Short-Term-Memory that sees the current game state and emits actions through several possible action heads.[38]
In 2018, OpenAI also trained a similar LSTM by policy gradients to control a human-like robot hand that manipulates physical objects with unprecedented dexterity.[39]
In 2019, DeepMind's program AlphaStar used a deep LSTM core to excel at the complex video game Starcraft.[40] This was viewed as significant progress towards Artificial General Intelligence.[40]
Applications of LSTM include:
qXi  Brad Smith, Microsoft’s president, last week told the Guardian that tech companies should stop behaving as though everything that is not illegal is acceptable. Mr Smith made a good argument that technology may be considered morally neutral but technologists can’t be. He is correct that software engineers ought to take much more seriously the moral consequences of their work. This argument operates on two levels: conscious and unconscious. It is easy to see the ethical issue that appeared to arise when, as a result of a series of its own confusing blog posts, Microsoft appeared to be selling facial recognition technology to US Immigration and Customs Enforcement while the Trump administration was separating children from parents at the US’s southern border. This was, as Microsoft later confirmed, false. The moral stance of more than 3,000 Google employees who protested about its Maven contract – where machine learning was to be used for military purposes, starting with drone imaging – with the US Department of Defense should be applauded. Google let the contract lapse. But people with different ethical viewpoints can take different views. In the case of the Maven contract, a rival with fewer qualms picked up the work. Much is contingent on public attitudes. Opinion polls show that Americans are not in favour of developing artificial intelligence technology for warfare, but this changes as soon as the country’s adversaries start to develop them. There is an economic aspect to be considered too. Shoshana Zuboff’s insight, that the exploitation of behavioural predictions covertly derived from the surveillance of users is capitalism’s latest stage, is key. What is our moral state when AI researchers are paid $1m a year but the people who label and classify the input data are paid $1.47 an hour. However, the most difficult human skills to replicate are the unconscious ones, the product of millennia of evolution. In AI this is known as Moravec’s paradox. “We are all prodigious Olympians in perceptual and motor areas, so good that we make the difficult look easy,” wrote the futurist Hans Moravec. It is these that our brains excel in, hidden but complex processes that machine learning attempts to replicate. This presents the maker of such technology with a unique problem of accountability. If a building falls, the authorities can investigate, spot a failure, and put it down to engineering. We absorb the lessons and hope to learn from our mistakes. But is this true for programmers? It is in the nature of AI that makers do not, and often cannot, predict what their creations do. We know how to make machines learn. But programmers do not understand completely the knowledge that intelligent computing acquires. If we did, we wouldn’t need computers to learn to learn. We’d know what they did and program the machines directly ourselves. They can recognise a face, a voice, be trained to judge people’s motivations or beat a computer game. But we cannot say exactly how. This is the genius and madness behind the technology. The promise of AI is that it will imbue machines with the ability to spot patterns from data, and make decisions faster and better than humans do. What happens if they make worse decisions faster? Governments need to pause and take stock of the societal repercussions of allowing machines over a few decades to replicate human skills that have been evolving for millions of years. But individuals and companies must take responsibility too. • This article was amended on 24 September 2019 to clarify that Microsoft had not sold facial recognition technology to US Immigration and Customs Enforcement.qXX  Bootstrap aggregating, also called bagging (from bootstrap aggregating), is a machine learning ensemble meta-algorithm designed to improve the stability and accuracy of machine learning algorithms used in statistical classification and regression. It also reduces variance and helps to avoid overfitting. Although it is usually applied to decision tree methods, it can be used with any type of method. Bagging is a special case of the model averaging approach.
Given a standard training set 



D


{\displaystyle D}

 of size n, bagging generates m new training sets 




D

i




{\displaystyle D_{i}}

, each of size n′, by sampling from D uniformly and with replacement. By sampling with replacement, some observations may be repeated in each 




D

i




{\displaystyle D_{i}}

. If n′=n, then for large n the set 




D

i




{\displaystyle D_{i}}

 is expected to have the fraction (1 - 1/e) (≈63.2%) of the unique examples of D, the rest being duplicates.[1] This kind of sample is known as a bootstrap sample. Then, m models  are fitted using the above m bootstrap samples and combined by averaging the output (for regression) or voting (for classification).
Bagging leads to "improvements for unstable procedures" (Breiman, 1996), which include, for example, artificial neural networks, classification and regression trees, and subset selection in linear regression (Breiman, 1994). An interesting application of bagging showing improvement in preimage learning is provided here.[2][3] On the other hand, it can mildly degrade the performance of stable methods such as K-nearest neighbors (Breiman, 1996).
To illustrate the basic principles of bagging, below is an analysis on the relationship between ozone and temperature (data from Rousseeuw and Leroy (1986), analysis done in R).
The relationship between temperature and ozone in this data set is apparently non-linear, based on the scatter plot. To mathematically describe this relationship, LOESS smoothers (with bandwidth 0.5) are used. 
Instead of building a single smoother from the complete data set, 100 bootstrap samples of the data were drawn. Each sample is different from the original data set, yet resembles it in distribution and variability. For each bootstrap sample, a LOESS smoother was fit. Predictions from these 100 smoothers were then made across the range of the data. The first 10 predicted smooth fits appear as grey lines in the figure below. The lines are clearly very wiggly and they overfit the data - a result of the bandwidth being too small.
By taking the average of 100 smoothers, each fitted to a subset of the original data set, we arrive at one bagged predictor (red line). Clearly, the mean is more stable and there is less overfit.
Bagging (Bootstrap aggregating) was proposed by Leo Breiman in 1994[4] to improve classification by combining classifications of randomly generated training sets.
qX;  Google DeepMind has announced its second collaboration with the NHS, working with Moorfields Eye Hospital in east London to build a machine learning system which will eventually be able to recognise sight-threatening conditions from just a digital scan of the eye. The collaboration is the second between the NHS and DeepMind, which is the artificial intelligence research arm of Google, but Deepmind’s co-founder, Mustafa Suleyman, says this is the first time the company is embarking purely on medical research. An earlier, ongoing, collaboration, with the Royal Free hospital in north London, is focused on direct patient care, using a smartphone app called Streams to monitor kidney function of patients. The Moorfields collaboration is also the first time DeepMind has used machine learning in a healthcare project. At the heart of the research is the sharing of a million anonymous eye scans, which the DeepMind researchers will use to train an algorithm to better spot the early signs of eye conditions such as wet age-related macular degeneration and diabetic retinopathy. Suleyman said: “There’s so much at stake, particularly with diabetic retinopathy. If you have diabetes you’re 25 times more likely to go blind. If we can detect this, and get in there as early as possible, then 98% of the most severe visual loss might be prevented.” Training a neural network to do the assessment of eye scans could vastly increase both the speed and accuracy of diagnosis, potentially saving the sight of thousands. The collaboration between the two organisations came about thanks to an unsolicited request from one doctor at Moorfields. Pearse Keane, a consultant ophthalmologist, contacted the Google subsidiary through its website to discuss the need to better analyse scans of the eye, and initiated the research project shortly after. “I’d been reading about deep learning and the success that technology had had in image recognition,” he said, when he came across an article about DeepMind training a machine to play Atari games – the company’s first public success. “I had the brainwave that deep learning could be really good at looking at the images of the eye. Optical Coherence Tomography is my area, and we have the largest depository of OCT images in the world. Within a couple of days I got in touch with Mustafa, and he replied.” DeepMind’s previous collaboration with the NHS had led to controversy, after it and its parter, the Royal Free hospital, were accused of not having the proper authority to share the records of patients who would be involved in the trial. At the time, Royal Free said that the arrangement “is the standard NHS information-sharing agreement set out by NHS England’s corporate information governance department and is the same as the other 1,500 agreements with third-party organisations that process NHS patient data.” Since the Moorfields collaboration involves anonymised information, the privacy hurdles are much lower. The company has been given permission for access through a research collaboration agreement with the hospital, and has published a research protocol, as is standard practice for medical trials. The company says the information shared amounts to “approximately 1m anonymous digital eye scans, along with some related anonymous information about eye condition and disease management. “This means it’s not possible to identify any individual patients from the scans. They’re also historic scans, meaning that while the results of our research may be used to improve future care, they won’t affect the care any patient receives today. The data used in this research is not personally identifiable. When research is working with such data, which is anonymous with no way for researchers to identify individual patients, explicit consent from patients for their data to be used in this way is not required.” Prof Peng Tee Khaw, the head of Moorfields’ ophthalmology research centre, said that the key to the collaboration was the huge increase in the volume of incredibly precise retinal scans available. “These scans are incredibly detailed, more detailed than any other scan of the body we do: we can see at the cellular level. But the problem for us is handling this amount of data. “It takes me my whole life experience to follow one patient’s history. And yet patients rely on my experience to predict their future. If we could use machine assisted deep learning, we could be so much better at doing this, because then I could have the experience of 10,000 lifetimes.” Somewhat oddly, the DeepMind/Moorfield collaboration is actually the second time that Google has looked at using machine learning to detect diabetic retinopathy in eye scans. An earlier, different, project was announced by Google chief executive Sundar Pichai onstage at the company’s annual developer conference, Google I/O, in May.qX�  Hacking, fraud and other clandestine online activities have been making headlines in recent weeks, giving rise to concerns that law enforcement agencies are losing the war against cybercriminals. But just how serious a threat to the public is cyber crime, and could data science hold the key to reversing the trend? RSA, the cyber security arm of US big data firm EMC, specialises in the use of advanced analytics and machine learning to predict and prevent online fraud. Its Anti-Fraud Command Centre (AFCC) has identified and terminated 500,000 such attacks in its eight year existence, half of which came in 2012 alone. This increasing detection rate is due in no small part to its rapid adoption of machine learning techniques. Five years ago RSA's Israeli operation undertook a step change, moving away from inflexible rule-based fraud detection systems, in favour of a self-improving approach using data science underpinned by Bayesian inferencing. In the UK, Detica - the data intelligence arm of BAE Systems - incorporates similar technologies into their cyber security efforts, in one case identifying advanced persistent threats (APTs) through data science-led methods that had previously gone unnoticed for 18 months by firms across the world. Every time a customer of one of RSA's clients makes a transaction using their online banking facility, 20 factors are recorded and fed automatically into the AFCC's database. These details are then combined into 150 fraud-risk features, each one consisting of a different mix of two or more of the 20 factors. For example, combining an IP and MAC address will give a better indication of whether a transaction is likely to be fraudulent or nor than either detail would give on its own. The features are then grouped into either prior indicators or bayesian predictors, based on the way in which they indicate fraudulent activity. Prior indicators are information that is indicative of fraud regardless of context, such as an initial failure to authenticate a transaction, while the predictors are data whose impact on fraud risk varies from one case to the next. In a given transaction each feature is assigned a risk score out of 100, with higher scores indicating a higher likelihood of fraud. All 150 scores are then combined using a set of algorithms, with the influence if any one score on the final total dependent on a unique weighting. The model's Bayesian nature arises from the fact that a predictor's weighting is calculated based on a constantly updating probability that, for a given customer, it is likely to indicate fraud. In this way, RSA's model becomes more accurate every time it spots a verified fraud event. It also means that if cybercriminals suddenly change tactic and an entirely new method of online theft emerges, the system will automatically detect the new risk pattern and incorporate it into the risk calculation for any event from that point on. One area where RSA is looking to improve their model is in the addition of new indicators to build on the 20 existing factors. "We constantly try to add more in this domain, done by explorative research and talking to customers", said Alon Kaufman, director of security analytics at the AFCC. "We're especially interested in mobile, where there are strong identifiers of device such as SIM card." Mobile phones are increasingly becoming the target of choice for cybercriminals, with a study by online security specialists Trend Micro identifying 350,000 malware threats aimed at mobiles in 2012 - up from 1,000 in 2011 - and forecasting one million in 2013. Android handsets are disproportionately likely to be affected, according to RSA's head of knowledge delivery, Daniel Cohen. "In 2012 around 70% of new smartphones were on an Android operating system, but 98% of mobile malware was targeted at Android., he said. One problem facing RSA, Detica and others is that sooner or later criminals will work out how to fool a system into thinking they are the person they are impersonating, by replicating the indicators their target would present. The challenge is to find data that is harder for a fraudster to copy. "We're looking increasingly at behavioural metrics - how you interact with your machine", said Kaufman. "Information such as typing speed, mouse movement and the order you access your bank's web pages are all strong indicators that you are who you say you are, but we can only incorporate these details into our model where our client provides them, and where they meet privacy regulations." The use of behavioural measures carries with it inherent risks concerning data protection law and privacy rights, and the issue of consent is increasingly cited. Almost four in five respondents to a 2012 Demos poll said their primary concern around personal data was that companies would use it without their permission - coming in just above worries about companies losing user data. Bridget Treacy, Head of UK privacy and information management practice at law firm Hunton & Williams said, "It might be possible to give individuals notice at the point at which their data are collected. In some circumstances notice may not be appropriate, but those circumstances should be limited, and clearly defined, with proper safeguards. Transparency and proportionality are key." "Inevitably there are tensions between organisations' use of data and individuals' privacy rights, and finding the right balance is not easy. However, the use of data for purposes such as the prevention and detection of fraud is not necessarily a zero sum game. It may be possible for some processing to take place utilising anonymised or pseudonymised data", she said. Detica operates a similar model to RSA, using intelligence gathering to guide its data-driven operations, and vice versa. "We will measure things such as a computer talking to a website that was registered two days ago, a computer talking to a website that no one else in that environment is talking to, or a laptop sending more data than it's receiving - which is more like server behaviour", said Richard Wilding, Detica's director of cyber security. "If you look for any of these activities individually, you'll be flooded with false positives, but we use big data methods to move from that initial trait identification to the bigger picture, and then our analysts will use the insights from that data to form their understanding of the attack", said Wilding. Following a data breach of its own in 2012, RSA now takes the same approach for its internal security, monitoring every time someone logs onto its internal network, their activity one inside and any extraction of data. "After the breach, I told them just bring me the data, and I'll tell you how we can prevent it next time. The data are different, the personnel are different, but the process is the same in terms of the underlying data science", said Yael Villa, site leader at RSA Israel. The model created by Villa's team flagged up retrospectively 30 potential sources of the breach, of which six were verified by EMC's global security officer as being events where an impersonator accessed the system in the digital guise of an employee. Full disclosure: earlier this month I spent a day at RSA's AFCC in Israel at the expense of EMC. Data journalism and data visualisations from the Guardian • Search the world's government data with our gateway • Search the world's global development data with our gateway • Flickr Please post your visualisations and mash-ups on our Flickr group• Contact us at data@theguardian.com • Get the A-Z of data• More at the Datastore directory• Follow us on TwitterqXh  A multilayer perceptron (MLP) is a class of feedforward artificial neural network (ANN). The term MLP is used ambiguously, sometimes loosely to refer to any feedforward ANN, sometimes strictly to refer to networks composed of multiple layers of perceptrons (with threshold activation); see § Terminology. Multilayer perceptrons are sometimes colloquially referred to as "vanilla" neural networks, especially when they have a single hidden layer.[1]
An MLP consists of at least three layers of nodes: an input layer, a hidden layer and an output layer. Except for the input nodes, each node is a neuron that uses a nonlinear activation function. MLP utilizes a supervised learning technique called backpropagation for training.[2][3] Its multiple layers and non-linear activation distinguish MLP from a linear perceptron. It can distinguish data that is not linearly separable.[4]
If a multilayer perceptron has a linear activation function in all neurons, that is, a linear function that maps the weighted inputs to the output of each neuron, then linear algebra shows that any number of layers can be reduced to a two-layer input-output model. In MLPs some neurons use a nonlinear activation function that was developed to model the frequency of action potentials, or firing, of biological neurons.
The two historically common activation functions are both sigmoids, and are described by
In recent developments of deep learning the rectifier linear unit (ReLU) is more frequently used as one of the possible ways to overcome the numerical problems related to the sigmoids.
The first is a hyperbolic tangent that ranges from -1 to 1, while the other is the logistic function, which is similar in shape but ranges from 0 to 1. Here 




y

i




{\displaystyle y_{i}}

 is the output of the 



i


{\displaystyle i}

th node (neuron) and 




v

i




{\displaystyle v_{i}}

 is the weighted sum of the input connections. Alternative activation functions have been proposed, including the rectifier and softplus functions. More specialized activation functions include radial basis functions (used in radial basis networks, another class of supervised neural network models).
The MLP consists of three or more layers (an input and an output layer with one or more hidden layers) of nonlinearly-activating nodes. Since MLPs are fully connected, each node in one layer connects with a certain weight 




w

i
j




{\displaystyle w_{ij}}

 to every node in the following layer.
Learning occurs in the perceptron by changing connection weights after each piece of data is processed, based on the amount of error in the output compared to the expected result. This is an example of supervised learning, and is carried out through backpropagation, a generalization of the least mean squares algorithm in the linear perceptron.
We can represent the degree of error in an output node 



j


{\displaystyle j}

 in the 



n


{\displaystyle n}

th data point (training example) by 




e

j


(
n
)
=

d

j


(
n
)
−

y

j


(
n
)


{\displaystyle e_{j}(n)=d_{j}(n)-y_{j}(n)}

, where 



d


{\displaystyle d}

 is the target value and 



y


{\displaystyle y}

 is the value produced by the perceptron. The node weights can then be adjusted based on corrections that minimize the error in the entire output, given by
Using gradient descent, the change in each weight is
where 




y

i




{\displaystyle y_{i}}

 is the output of the previous neuron and 



η


{\displaystyle \eta }

 is the learning rate, which is selected to ensure that the weights quickly converge to a response, without oscillations.
The derivative to be calculated depends on the induced local field 




v

j




{\displaystyle v_{j}}

, which itself varies. It is easy to prove that for an output node this derivative can be simplified to
where 




ϕ

′




{\displaystyle \phi ^{\prime }}

 is the derivative of the activation function described above, which itself does not vary. The analysis is more difficult for the change in weights to a hidden node, but it can be shown that the relevant derivative is
This depends on the change in weights of the 



k


{\displaystyle k}

th nodes, which represent the output layer. So to change the hidden layer weights, the output layer weights change according to the derivative of the activation function, and so this algorithm represents a backpropagation of the activation function.[5]

The term "multilayer perceptron" does not refer to a single perceptron that has multiple layers. Rather, it contains many perceptrons that are organized into layers. An alternative is "multilayer perceptron network". Moreover, MLP "perceptrons" are not perceptrons in the strictest possible sense. True perceptrons are formally a special case of artificial neurons that use a threshold activation function such as the Heaviside step function. MLP perceptrons can employ arbitrary activation functions. A true perceptron performs binary classification (either this or that), an MLP neuron is free to either perform classification or regression, depending upon its activation function.
The term "multilayer perceptron" later was applied without respect to nature of the nodes/layers, which can be composed of arbitrarily defined artificial neurons, and not perceptrons specifically. This interpretation avoids the loosening of the definition of "perceptron" to mean an artificial neuron in general.
MLPs are useful in research for their ability to solve problems stochastically, which often allows approximate solutions for extremely complex problems like fitness approximation.
MLPs are universal function approximators as shown by Cybenko's theorem,[4] so they can be used to create mathematical models by regression analysis. As classification is a particular case of regression when the response variable is categorical, MLPs make good classifier algorithms.
MLPs were a popular machine learning solution in the 1980s, finding applications in diverse fields such as speech recognition, image recognition, and machine translation software,[6] but thereafter faced strong competition from much simpler (and related[7]) support vector machines. Interest in backpropagation networks returned due to the successes of deep learning.
qX�  
TensorFlow is a free and open-source software library for dataflow and differentiable programming across a range of tasks. It is a symbolic math library, and is also used for machine learning applications such as neural networks.[4] It is used for both research and production at Google.‍[4]:min 0:15/2:17 [5]:p.2 [4]:0:26/2:17
TensorFlow was developed by the Google Brain team for internal Google use. It was released under the Apache License 2.0 on November 9, 2015.[1][6]
Starting in 2011, Google Brain built DistBelief as a proprietary machine learning system based on deep learning neural networks. Its use grew rapidly across diverse Alphabet companies in both research and commercial applications.[5][7] Google assigned multiple computer scientists, including Jeff Dean, to simplify and refactor the codebase of DistBelief into a faster, more robust application-grade library, which became TensorFlow.[8] In 2009, the team, led by Geoffrey Hinton, had implemented generalized backpropagation and other improvements which allowed generation of neural networks with substantially higher accuracy, for instance a 25% reduction in errors in speech recognition.[9]
TensorFlow is Google Brain's second-generation system. Version 1.0.0 was released on February 11, 2017.[10]  While the reference implementation runs on single devices, TensorFlow can run on multiple CPUs and GPUs (with optional CUDA and SYCL extensions for general-purpose computing on graphics processing units).[11] TensorFlow is available on 64-bit Linux, macOS, Windows, and mobile computing platforms including Android and iOS.
Its flexible architecture allows for the easy deployment of computation across a variety of platforms (CPUs, GPUs, TPUs), and from desktops to clusters of servers to mobile and edge devices.
TensorFlow computations are expressed as stateful dataflow graphs. The name TensorFlow derives from the operations that such neural networks perform on multidimensional data arrays, which are referred to as tensors. During the Google I/O Conference in June 2016, Jeff Dean stated that 1,500 repositories on GitHub mentioned TensorFlow, of which only 5 were from Google.[12]
In December 2017, developers from Google, Cisco, RedHat, CoreOS, and CaiCloud introduced Kubeflow at a conference. Kubeflow allows operation and deployment of TensorFlow on Kubernetes.
In March 2018, Google announced TensorFlow.js version 1.0 for machine learning in JavaScript.[13]
In Jan 2019, Google announced TensorFlow 2.0.[14] It became officially available in Sep 2019.[15]
In May 2019, Google announced TensorFlow Graphics for deep learning in computer graphics.[16]
In May 2016, Google announced its Tensor processing unit (TPU), an application-specific integrated circuit (a hardware chip) built specifically for machine learning and tailored for TensorFlow. TPU is a programmable AI accelerator designed to provide high throughput of low-precision arithmetic (e.g., 8-bit), and oriented toward using or running models rather than training them. Google announced they had been running TPUs inside their data centers for more than a year, and had found them to deliver an order of magnitude better-optimized performance per watt for machine learning.[17]
In May 2017, Google announced the second-generation, as well as the availability of the TPUs in Google Compute Engine.[18] The second-generation TPUs deliver up to 180 teraflops of performance, and when organized into clusters of 64 TPUs, provide up to 11.5 petaflops.
In May 2018, Google announced the third-generation TPUs delivering up to 420 teraflops of performance and 128 GB high bandwidth memory (HBM). Cloud TPU v3 Pods offer 100+ petaflops of performance and 32 TB HBM.[19]
In February 2018, Google announced that they were making TPUs available in beta on the Google Cloud Platform.[20]
In July 2018, the Edge TPU was announced. Edge TPU is Google’s purpose-built ASIC chip designed to run TensorFlow Lite machine learning (ML) models on small client computing devices such as smartphones[21] known as edge computing.
In May 2017, Google announced a software stack specifically for mobile development, TensorFlow Lite.[22]  In January 2019, TensorFlow team released a developer preview of the mobile GPU inference engine with OpenGL ES 3.1 Compute Shaders on Android devices and Metal Compute Shaders on iOS devices.[23] In May 2019, Google announced that their TensorFlow Lite Micro (also known as TensorFlow Lite for Microcontrollers) and ARM's uTensor would be merging.[24]
In October 2017, Google released the Google Pixel 2 which featured their Pixel Visual Core (PVC), a fully programmable image, vision and AI processor for mobile devices. The PVC supports TensorFlow for machine learning (and Halide for image processing).
Google officially released RankBrain on October 26, 2015, backed by TensorFlow.
Google also released Colaboratory, which is a TensorFlow Jupyter notebook environment that requires no setup to use.[25]
On March 1, 2018, Google released its Machine Learning Crash Course (MLCC). Originally designed to help equip Google employees with practical artificial intelligence and machine learning fundamentals, Google rolled out its free TensorFlow workshops in several cities around the world before finally releasing the course to the public.[26]
TensorFlow provides stable Python (for version 3.7 across all platforms)[27] and C APIs;[28] and without API backwards compatibility guarantee: C++, Go, Java,[29] JavaScript[3] and Swift (early release).[30][31]  Third-party packages are available for C#,[32][33] Haskell,[34] Julia,[35] MATLAB,[36] R,[37] Scala,[38] Rust,[39] OCaml,[40] and Crystal.[41]
"New language support should be built on top of the C API. However, [..] not all functionality is available in C yet."[42] Some more functionality is provided by the Python API.
Among the applications for which TensorFlow is the foundation, are automated image-captioning software, such as DeepDream.[43] RankBrain now handles a substantial number of search queries, replacing and supplementing traditional static algorithm-based search results.[44]
qX�  
Geoffrey Everest Hinton CC FRS FRSC[11] (born 6 December 1947) is an English Canadian cognitive psychologist and computer scientist, most noted for his work on artificial neural networks. Since 2013 he divides his time working for Google (Google Brain) and the University of Toronto. In 2017, he cofounded and became the Chief Scientific Advisor of the Vector Institute in Toronto.  [12][13]
With David E. Rumelhart and Ronald J. Williams, Hinton was co-author of a highly cited paper published in 1986 that popularized the backpropagation algorithm for training multi-layer neural networks,[14] although they were not the first to propose the approach.[15] Hinton is viewed by some as a leading figure in the deep learning community and is referred to by some as the "Godfather of Deep Learning".[16][17][18][19][20] The dramatic image-recognition milestone of the AlexNet designed by his student Alex Krizhevsky[21] for the ImageNet challenge 2012[22] helped to revolutionize the field of computer vision.[23] Hinton was awarded the 2018 Turing Prize alongside Yoshua Bengio and Yann LeCun for their work on deep learning.[24]
Hinton - together with Yoshua Bengio and Yann LeCun - are referred to by some as the "Godfathers of AI" and "Godfathers of Deep Learning". [25][26][27][28][29][30][31]
Hinton was educated at King's College, Cambridge graduating in 1970, with a Bachelor of Arts in experimental psychology.[1] He continued his study at the University of Edinburgh where he was awarded a PhD in artificial intelligence in 1978 for research supervised by Christopher Longuet-Higgins.[3][32]
After his PhD he worked at the University of Sussex, and (after difficulty finding funding in Britain)[33] the University of California, San Diego, and Carnegie Mellon University.[1] He was the founding director of the Gatsby Charitable Foundation Computational Neuroscience Unit at University College London,[1] and is currently[update][34] a professor in the computer science department at the University of Toronto. He holds a Canada Research Chair in Machine Learning, and is currently an advisor for the Learning in Machines & Brains program at the Canadian Institute for Advanced Research. Hinton taught a free online course on Neural Networks on the education platform Coursera in 2012.[35] Hinton joined Google in March 2013 when his company, DNNresearch Inc., was acquired. He is planning to "divide his time between his university research and his work at Google".[36]
Hinton's research investigates ways of using neural networks for machine learning, memory, perception and symbol processing. He has authored or co-authored over 200 peer reviewed publications.[2][37]
While Hinton was a professor at Carnegie Mellon University (1982–1987), David E. Rumelhart and Hinton and Ronald J. Williams applied the backpropagation algorithm to multi-layer neural networks. Their experiments showed that such networks can learn useful internal representations of data.[14]  In an interview of 2018,[38] Hinton said that "David E. Rumelhart came up with the basic idea of backpropagation, so it's his invention." Although this work was important in popularizing backpropagation, it was not the first to suggest the approach.[15] Reverse-mode automatic differentiation, of which backpropagation is a special case, was proposed by Seppo Linnainmaa in 1970, and Paul Werbos proposed to use it to train neural networks in 1974.[15]
During the same period, Hinton co-invented Boltzmann machines with David Ackley and Terry Sejnowski.[39] His other contributions to neural network research include distributed representations, time delay neural network, mixtures of experts, Helmholtz machines and Product of Experts. In 2007 Hinton coauthored an unsupervised learning paper titled Unsupervised learning of image transformations.[40] An accessible introduction to Geoffrey Hinton's research can be found in his articles in Scientific American in September 1992 and October 1993..[41]
In October and November 2017 respectively, Hinton published two open access research papers[42][43] on the theme of capsule neural networks, which according to Hinton are "finally something that works well."[44]
Notable former PhD students and postdoctoral researchers from his group include Richard Zemel,[3][6] Brendan Frey,[7] Radford M. Neal,[8] Ruslan Salakhutdinov,[9] Ilya Sutskever,[10] Yann LeCun[45] and Zoubin Ghahramani.
Hinton was elected a Fellow of the Royal Society (FRS) in 1998.[11] He was the first winner of the Rumelhart Prize in 2001.[46] His certificate of election for the Royal Society reads: 
In 2001, Hinton was awarded an Honorary Doctorate from the University of Edinburgh.[48] He was the 2005 recipient of the IJCAI Award for Research Excellence lifetime-achievement award.[citation needed] He has also been awarded the 2011 Herzberg Canada Gold Medal for Science and Engineering.[49] In 2013, Hinton was awarded an Honorary Doctorate from the Université de Sherbrooke.[citation needed]
In 2016, he was elected a foreign member of National Academy of Engineering "For contributions to the theory and practice of artificial neural networks and their application to speech recognition and computer vision".[50] He also received the 2016 IEEE/RSE Wolfson James Clerk Maxwell Award.[51]
He has won the BBVA Foundation Frontiers of Knowledge Award (2016) in the Information and Communication Technologies category "for his pioneering and highly influential work" to endow machines with the ability to learn.[citation needed]
Together with Yann LeCun, and Yoshua Bengio, Hinton won the 2018 Turing Award for conceptual and engineering breakthroughs that have made deep neural networks a critical component of computing.[52][53][54]
Hinton is the great-great-grandson both of logician George Boole whose work eventually became one of the foundations of modern computer science, and of surgeon and author James Hinton.[55] who was the father of Charles Howard Hinton. Hinton's father was Howard Hinton.[1][56] His middle name is from another relative, George Everest.[33] He is the nephew of the economist Colin Clark.[57]  He lost his first wife to ovarian cancer in 1994.[58]
Hinton moved from the U.S. to Canada in part due to disillusionment with Ronald Reagan-era politics and disapproval of military funding of artificial intelligence.[33]
Hinton has petitioned against lethal autonomous weapons. Regarding existential risk from artificial intelligence, Hinton typically declines to make predictions more than five years into the future, noting that exponential progress makes the uncertainty too great.[59] However, in an informal conversation with the AI risk researcher Nick Bostrom in November 2015, overheard by journalist Raffi Khatchadourian,[60] he is reported to have stated that he did not expect general A.I. to be achieved for decades (“No sooner than 2070”), and that, in the context of a dichotomy earlier introduced by Bostrom between people who think managing existential risk from artificial intelligence is probably hopeless versus easy enough that it will be solved automatically, Hinton "[is] in the camp that is hopeless.”[60] He has stated, “I think political systems will use it to terrorize people” and has expressed his belief that agencies like the National Security Agency (NSA) are already attempting to abuse similar technology.[60]
Asked by Nick Bostrom why he continues research despite his grave concerns, Hinton stated, "I could give you the usual arguments. But the truth is that the prospect of discovery is too sweet."[60]
According to the same report, Hinton does not categorically rule out human beings controlling an artificial superintelligence, but warns that "there is not a good track record of less intelligent things controlling things of greater intelligence".[60]
"All text published under the heading 'Biography' on Fellow profile pages is available under Creative Commons Attribution 4.0 International License." --"Royal Society Terms, conditions and policies". Archived from the original on 11 November 2016. Retrieved 9 March 2016.CS1 maint: BOT: original-url status unknown (link)
qXN  Statistical relational learning (SRL) is a subdiscipline of artificial intelligence and machine learning that is concerned with domain models that exhibit both uncertainty (which can be dealt with using statistical methods) and complex, relational structure.[1][2] Note that SRL is sometimes called Relational Machine Learning (RML) in the literature. Typically, the knowledge representation formalisms developed in SRL use (a subset of) first-order logic to describe relational properties of a domain in a general manner (universal quantification) and draw upon probabilistic graphical models (such as Bayesian networks or Markov networks) to model the uncertainty; some also build upon the methods of inductive logic programming. Significant contributions to the field have been made since the late 1990s.[3]
As is evident from the characterization above, the field is not strictly limited to learning aspects; it is equally concerned with reasoning (specifically probabilistic inference) and knowledge representation. Therefore, alternative terms that reflect the main foci of the field include statistical relational learning and reasoning (emphasizing the importance of reasoning) and first-order probabilistic languages (emphasizing the key properties of the languages with which models are represented).
A number of canonical tasks are associated with statistical relational learning, the most common ones being[4]
One of the fundamental design goals of the representation formalisms developed in SRL is to abstract away from concrete entities and to represent instead general principles that are intended to be universally applicable. Since there are countless ways in which such principles can be represented, many representation formalisms have been proposed in recent years.[1] In the following, some of the more common ones are listed in alphabetical order:
qX}  ”Any sufficiently advanced technology,” wrote the sci-fi eminence grise Arthur C Clarke, “is indistinguishable from magic.” This quotation, endlessly recycled by tech boosters, is possibly the most pernicious utterance Clarke ever made because it encourages hypnotised wonderment and disables our critical faculties. For if something is “magic” then by definition it is inexplicable. There’s no point in asking questions about it; just accept it for what it is, lie back and suspend disbelief. Currently, the technology that most attracts magical thinking is artificial intelligence (AI). Enthusiasts portray it as the most important thing since the invention of the wheel. Pessimists view it as an existential threat to humanity: the first “superintelligent” machine we build will be the beginning of the end for humankind; the only question thereafter will be whether smart machines will keep us as pets. In both cases there seems to be an inverse correlation between the intensity of people’s convictions about AI and their actual knowledge of the technology. The experts seem calmly sanguine, while the boosters seem blissfully unaware that the artificial “intelligence” they extol is actually a relatively mundane combination of machine learning (ML) plus big data. ML uses statistical techniques to give computers the ability to “learn” – ie use data to progressively improve performance on a specific task, without being explicitly programmed. A machine-learning system is a bundle of algorithms that take in torrents of data at one end and spit out inferences, correlations, recommendations and possibly even decisions at the other end. And the technology is already ubiquitous: virtually every interaction we have with Google, Amazon, Facebook, Netflix, Spotify et al is mediated by machine-learning systems. It’s even got to the point where one prominent AI guru, Andrew Ng, likens ML to electricity. To many corporate executives, a machine that can learn more about their customers than they ever knew seems magical. Think, for example, of the moment Walmart discovered that among the things their US customers stocked up on before a hurricane warning – apart from the usual stuff – were beer and strawberry Pop-Tarts! Inevitably, corporate enthusiasm for the magical technology soon spread beyond supermarket stock-controllers to public authorities. Machine learning rapidly found its way into traffic forecasting, “predictive” policing (in which ML highlights areas where crime is “more likely”), decisions about prisoner parole, and so on. Among the rationales for this feeding frenzy are increased efficiency, better policing, more “objective” decision-making and, of course, providing more responsive public services. This “mission creep” has not gone unnoticed. Critics have pointed out that the old computing adage “garbage in, garbage out” also applies to ML. If the data from which a machine “learns” is biased, then the outputs will reflect those biases. And this could become generalised: we may have created a technology that – however good it is at recommending films you might like – may actually morph into a powerful amplifier of social, economic and cultural inequalities. In all of this sociopolitical criticism of ML, however, what has gone unchallenged is the idea that the technology itself is technically sound – in other words that any problematic outcomes it produces are, ultimately, down to flaws in the input data. But now it turns out that this comforting assumption may also be questionable. At the most recent Nips (Neural Information Processing Systems) conference – the huge annual gathering of ML experts – Ali Rahimi, one of the field’s acknowledged stars, lobbed an intellectual grenade into the audience. In a remarkable lecture he likened ML to medieval alchemy. Both fields worked to a certain extent – alchemists discovered metallurgy and glass-making; ML researchers have built machines that can beat human Go champions and identify objects from pictures. But just as alchemy lacked a scientific basis, so, argued Rahimi, does ML. Researchers, he claimed, often can’t explain the inner workings of their mathematical models: they lack rigorous theoretical understandings of their tools and in that sense are currently operating in alchemical rather than scientific mode. Does this matter? Emphatically yes. As Rahimi puts it: “We are building systems that govern healthcare and mediate our civic dialogue. We would influence elections. I would like to live in a society whose systems are built on top of verifiable, rigorous, thorough knowledge, and not on alchemy.” Me too. We built what we like to call a civilisation on electricity. But at least we understood why and how it worked. If Rahimi is right, then we’re nowhere near that with AI – yet. So let’s take a break from magical thinking about it. Peter Thiel speaks The billionaire entrepreneur may be a pain, but he’s an interesting one – this interview in Swiss magazine Die Weltwoche shows why. Share, commentThough under the cosh of public opinion at the moment, Facebook is, by any corporate criterion, a remarkably strong company. Ben Thompson explains why in an uncomfortable but persuasive essay. Goop dreamsKnow nothing about Gwyneth Paltrow and her Goop company? Me too. But then I stumbled on Taffy Brodesser-Akner’s long New York Times article and was hooked: not on Paltrow; on the essay.qX#  An artificial intelligence tool that has revolutionised the ability of computers to interpret everyday language has been shown to exhibit striking gender and racial biases. The findings raise the spectre of existing social inequalities and prejudices being reinforced in new and unpredictable ways as an increasing number of decisions affecting our everyday lives are ceded to automatons.  In the past few years, the ability of programs such as Google Translate to interpret language has improved dramatically. These gains have been thanks to new machine learning techniques and the availability of vast amounts of online text data, on which the algorithms can be trained. However, as machines are getting closer to acquiring human-like language abilities, they are also absorbing the deeply ingrained biases concealed within the patterns of language use, the latest research reveals. Joanna Bryson, a computer scientist at the University of Bath and a co-author, said: “A lot of people are saying this is showing that AI is prejudiced. No. This is showing we’re prejudiced and that AI is learning it.” But Bryson warned that AI has the potential to reinforce existing biases because, unlike humans, algorithms may be unequipped to consciously counteract learned biases. “A danger would be if you had an AI system that didn’t have an explicit part that was driven by moral ideas, that would be bad,” she said. The research, published in the journal Science, focuses on a machine learning tool known as “word embedding”, which is already transforming the way computers interpret speech and text. Some argue that the natural next step for the technology may involve machines developing human-like abilities such as common sense and logic. “A major reason we chose to study word embeddings is that they have been spectacularly successful in the last few years in helping computers make sense of language,” said Arvind Narayanan, a computer scientist at Princeton University and the paper’s senior author.  The approach, which is already used in web search and machine translation, works by building up a mathematical representation of language, in which the meaning of a word is distilled into a series of numbers (known as a word vector) based on which other words most frequently appear alongside it. Perhaps surprisingly, this purely statistical approach appears to capture the rich cultural and social context of what a word means in the way that a dictionary definition would be incapable of. For instance, in the mathematical “language space”, words for flowers are clustered closer to words linked to pleasantness, while words for insects are closer to words linked to unpleasantness, reflecting common views on the relative merits of insects versus flowers. The latest paper shows that some more troubling implicit biases seen in human psychology experiments are also readily acquired by algorithms. The words “female” and “woman” were more closely associated with arts and humanities occupations and with the home, while “male” and “man” were closer to maths and engineering professions. And the AI system was more likely to associate European American names with pleasant words such as “gift” or “happy”, while African American names were more commonly associated with unpleasant words. The findings suggest that algorithms have acquired the same biases that lead people (in the UK and US, at least) to match pleasant words and white faces in implicit association tests.  These biases can have a profound impact on human behaviour. One previous study showed that an identical CV is 50% more likely to result in an interview invitation if the candidate’s name is European American than if it is African American. The latest results suggest that algorithms, unless explicitly programmed to address this, will be riddled with the same social prejudices. “If you didn’t believe that there was racism associated with people’s names, this shows it’s there,” said Bryson. The machine learning tool used in the study was trained on a dataset known as the “common crawl” corpus – a list of 840bn words that have been taken as they appear from material published online. Similar results were found when the same tools were trained on data from Google News. Sandra Wachter, a researcher in data ethics and algorithms at the University of Oxford, said: “The world is biased, the historical data is biased, hence it is not surprising that we receive biased results.” Rather than algorithms representing a threat, they could present an opportunity to address bias and counteract it where appropriate, she added. “At least with algorithms, we can potentially know when the algorithm is biased,” she said. “Humans, for example, could lie about the reasons they did not hire someone. In contrast, we do not expect algorithms to lie or deceive us.” However, Wachter said the question of how to eliminate inappropriate bias from algorithms designed to understand language, without stripping away their powers of interpretation, would be challenging. “We can, in principle, build systems that detect biased decision-making, and then act on it,” said Wachter, who along with others has called for an AI watchdog to be established. “This is a very complicated task, but it is a responsibility that we as society should not shy away from.”qX!  The tech craze du jour is machine learning (ML). Billions of dollars of venture capital are being poured into it. All the big tech companies are deep into it. Every computer science student doing a PhD on it is assured of lucrative employment after graduation at his or her pick of technology companies. One of the most popular courses at Stanford is CS229: Machine Learning. Newspapers and magazines extol the wonders of the technology. ML is the magic sauce that enables Amazon to know what you might want to buy next, and Netflix to guess which films might interest you, given your recent viewing history. To non-geeks, ML is impenetrable, and therefore intimidating. Exotic terminology abounds: neural networks, backpropagation, random forests, Bayesian networks, quadratic classifiers – that sort of thing. Accordingly, a kind of high priesthood has assembled around the technology which, like all priesthoods, tends to patronise anyone who wonders whether this arcane technology might not be, well, entirely good for humanity. “Don’t you worry about a thing, dear,” is the general tone. “We know what we’re doing.” When I mentioned ML to a classicist friend of mine recently, he replied: “What, exactly, is the machine learning?” That turns out to be the key question. Machine learning, you see, is best understood as a giant computer-powered sausage-making machine. Into the machine is fed a giant helping of data (called a training set) and, after a bit of algorithmic whirring, out comes the sausage – in the shape of a correlation or a pattern that the algorithm has “learned” from the training set. The fact the same generic approach works across a range of domains should make you suspicious The machine is then fed a new datastream, and on the basis of what it has “learned”, proceeds to emit correlations, recommendations and perhaps even judgments (such as: this person is likely to reoffend if granted parole; or that person should be granted a loan). And because these outputs are computer-generated, they are currently regarded with awe and amazement by bemused citizens who are not privy to the aforesaid algorithmic magic. It’s time to wean ourselves off this servile cringe. A good place to begin would be to start using everyday metaphors for all this exotic gobbledegook. Cue Maciej Cegłowski, who describes himself as “a painter and computer guy” who lives in San Francisco and maintains one of the most delightful blogs to be found on the web. Last month, Cegłowski was invited to give a talk at the US Library of Congress in which he proposed a novel metaphor. “Machine learning,” he says, “is like a deep-fat fryer. If you’ve never deep-fried something before, you think to yourself: ‘This is amazing! I bet this would work on anything!’ And it kind of does. In our case, the deep fryer is a toolbox of statistical techniques. The names keep changing – it used to be unsupervised learning, now it’s called big data or deep learning or AI. Next year it will be called something else. But the core ideas don’t change. You train a computer on lots of data, and it learns to recognise structure.” “But,” continues Cegłowski, “the fact that the same generic approach works across a wide range of domains should make you suspicious about how much insight it’s adding. In any deep-frying situation, a good question to ask is: what is this stuff being fried in?” The cooking oil, in the case of machine learning, is the data used for training. If the data is contaminated – by error, selectivity or bias – so too will be the patterns learned by the software. And of course, the ML priesthood knows that, so the more conscientious practitioners go to considerable lengths to try to detect and correct for biased results in applications of the technology. For an increasing number of ML applications, though, the training sets are just huge collections of everyday conversations – culled, for example, from social media. That sounds OK: after all, ordinary speech is just that. But a remarkable piece of research by AI researchers at Princeton and the University of Bath reveals that even everyday speech has embedded biases of which most of us are unaware. “Language itself contains recoverable and accurate imprints of our historic biases,” they write, “whether these are morally neutral as towards insects or flowers, problematic as towards race or gender, or even simply veridical, reflecting the status quo for the distribution of gender with respect to careers or first names.” And of course these hidden biases are inevitably captured by machine learning. I suspect that Wittgenstein would have loved this research: it confirms his belief that the meaning of a word is not to be found in some abstract definition, but in its use in everyday language. Maybe ML geeks should read his Tractatus.qX  Vowpal Wabbit (also known as "VW") is an open-source fast online interactive machine learning system library and program developed originally at  Yahoo! Research, and currently at Microsoft Research.  It was started and is led by John Langford. Vowpal Wabbit's interactive learning support is particularly notable including Contextual Bandits, Active Learning, and forms of guided Reinforcement Learning.  Vowpal Wabbit provides an efficient scalable out-of-core implementation with support for a number of machine learning reductions, importance weighting, and a selection of different loss functions and optimization algorithms.
The VW program supports:
Vowpal wabbit has been used to learn a tera-feature (1012) data-set on 1000 nodes in one hour.[1] Its scalability is aided by several factors:
qXg  Learning classifier systems, or LCS, are a paradigm of rule-based machine learning methods that combine a discovery component (e.g. typically a genetic algorithm) with a learning component (performing either supervised learning, reinforcement learning, or unsupervised learning).[2]  Learning classifier systems seek to identify a set of context-dependent rules that collectively store and apply knowledge in a piecewise manner in order to make predictions (e.g. behavior modeling,[3] classification,[4][5] data mining,[5][6][7] regression,[8] function approximation,[9] or game strategy).  This approach allows complex solution spaces to be broken up into smaller, simpler parts.
The founding concepts behind learning classifier systems came from attempts to model complex adaptive systems, using rule-based agents to form an artificial cognitive system (i.e. artificial intelligence).
The architecture and components of a given learning classifier system can be quite variable.  It is useful to think of an LCS as a machine consisting of several interacting components.  Components may be added or removed, or existing components modified/exchanged to suit the demands of a given problem domain (like algorithmic building blocks) or to make the algorithm flexible enough to function in many different problem domains.  As a result, the LCS paradigm can be flexibly applied to many problem domains that call for machine learning.  The major divisions among LCS implementations are as follows: (1) Michigan-style architecture vs. Pittsburgh-style architecture[10], (2) reinforcement learning vs. supervised learning, (3) incremental learning vs. batch learning, (4) online learning vs. offline learning, (5) strength-based fitness vs. accuracy-based fitness, and (6) complete action mapping vs best action mapping.   These divisions are not necessarily mutually exclusive. For example, XCS,[11] the best known and best studied LCS algorithm, is Michigan-style, was designed for reinforcement learning but can also perform supervised learning, applies incremental learning that can be either online or offline, applies accuracy-based fitness, and seeks to generate a complete action mapping.
Keeping in mind that LCS is a paradigm for genetic-based machine learning rather than a specific method, the following outlines key elements of a generic, modern (i.e. post-XCS) LCS algorithm.  For simplicity let us focus on Michigan-style architecture with supervised learning.  See the illustrations  on the right laying out the sequential steps involved in this type of generic LCS.
The environment is the source of data upon which an LCS learns.  It can be an offline, finite training dataset (characteristic of a data mining, classification, or regression problem), or an online sequential stream of live training instances.  Each training instance is assumed to include some number of features (also referred to as attributes, or independent variables), and a single endpoint of interest (also referred to as the class, action, phenotype, prediction, or dependent variable).  Part of LCS learning can involve feature selection, therefore not all of the features in the training data need be informative.  The set of feature values of an instance is commonly referred to as the state.  For simplicity let's assume an example problem domain with Boolean/binary features and a Boolean/binary class.  For Michigan-style systems, one instance from the environment is trained on each learning cycle (i.e. incremental learning).  Pittsburgh-style systems perform batch learning, where rule-sets are evaluated each iteration over much or all of the training data.
A rule is a context dependent relationship between state values and some prediction.  Rules typically take the form of an {IF:THEN} expression, (e.g.  {IF 'condition' THEN 'action'}, or as a more specific example, {IF 'red' AND 'octagon' THEN 'stop-sign'}).   A critical concept in LCS and rule-based machine learning alike, is that an individual rule is not in itself a model, since the rule is only applicable when its condition is satisfied.  Think of a rule as a "local-model" of the solution space.
Rules can be represented in many different ways to handle different data types (e.g. binary, discrete-valued, ordinal, continuous-valued).  Given binary data LCS traditionally applies a ternary rule representation (i.e. rules can include either a 0, 1, or '#' for each feature in the data).  The 'don't care' symbol (i.e. '#') serves as a wild card within a rule's condition allowing rules, and the system as a whole to generalize relationships between features and the target endpoint to be predicted. Consider the following rule (#1###0 ~ 1) (i.e. condition ~ action).  This rule can be interpreted as: IF the second feature = 1 AND the sixth feature = 0 THEN the class prediction = 1.  We would say that the second and sixth features were specified in this rule, while the others were generalized. This rule, and the corresponding prediction are only applicable to an instance when the condition of the rule is satisfied by the instance.  This is more commonly referred to as matching.  In Michigan-style LCS, each rule has its own fitness, as well as a number of other rule-parameters associated with it that can describe the number of copies of that rule that exist (i.e. the numerosity), the age of the rule, its accuracy, or the accuracy of its reward predictions, and other descriptive or experiential statistics.  A rule along with its parameters is often referred to as a classifier.  In Michigan-style systems, classifiers are contained within a population [P] that has a user defined maximum number of classifiers.  Unlike most stochastic search algorithms (e.g. evolutionary algorithms), LCS populations start out empty (i.e. there is no need to randomly initialize a rule population).  Classifiers will instead be initially introduced to the population with a covering mechanism.
In any LCS, the trained model is a set of rules/classifiers, rather than any single rule/classifier.  In Michigan-style LCS, the entire trained (and optionally, compacted) classifier population forms the prediction model.
One of the most critical and often time-consuming elements of an LCS is the matching process.  The first step in an LCS learning cycle takes a single training instance from the environment and passes it to [P] where matching takes place.  In step two, every rule in [P] is now compared to the training instance to see which rules match (i.e. are contextually relevant to the current instance).  In step three, any matching rules are moved to a match set [M].  A rule matches a training instance if all feature values specified in the rule condition are equivalent to the corresponding feature value in the training instance.  For example, assuming the training instance is (001001 ~ 0), these rules would match: (###0## ~ 0), (00###1 ~ 0), (#01001 ~ 1), but these rules would not (1##### ~ 0), (000##1 ~ 0), (#0#1#0 ~ 1).  Notice that in matching, the endpoint/action specified by the rule is not taken into consideration.  As a result, the match set may contain classifiers that propose conflicting actions.  In the fourth step, since we are performing supervised learning, [M] is divided into a correct set [C] and an incorrect set [I].  A matching rule goes into the correct set if it proposes the correct action (based on the known action of the training instance), otherwise it goes into [I].  In reinforcement learning LCS, an action set [A] would be formed here instead, since the correct action is not known.
At this point in the learning cycle, if no classifiers made it into either [M] or [C] (as would be the case when the population starts off empty), the covering mechanism is applied (fifth step).  Covering is a form of online smart population initialization. Covering randomly generates a rule that matches the current training instance (and in the case of supervised learning, that rule is also generated with the correct action.  Assuming the training instance is (001001 ~ 0), covering might generate any of the following rules:  (#0#0## ~ 0), (001001 ~ 0), (#010## ~ 0).  Covering not only ensures that each learning cycle there is at least one correct, matching rule in [C], but that any rule initialized into the population will match at least one training instance.  This prevents LCS from exploring the search space of rules that do not match any training instances.
In the sixth step, the rule parameters of any rule in [M] are updated to reflect the new experience gained from the current training instance.  Depending on the LCS algorithm, a number of updates can take place at this step.  For supervised learning, we can simply update the accuracy/error of a rule.  Rule accuracy/error is different than model accuracy/error, since it is not calculated over the entire training data, but only over all instances that it matched.  Rule accuracy is calculated by dividing the number of times the rule was in a correct set [C] by the number of times it was in a match set [M].  Rule accuracy can be thought of as a 'local accuracy'.  Rule fitness is also updated here, and is commonly calculated as a function of rule accuracy.  The concept of fitness is taken directly from classic genetic algorithms.  Be aware that there are many variations on how LCS updates parameters in order to perform credit assignment and learning.
In the seventh step, a subsumption mechanism is typically applied.  Subsumption is an explicit generalization mechanism that merges classifiers that cover redundant parts of the problem space.  The subsuming classifier effectively absorbs the subsumed classifier (and has its numerosity increased).  This can only happen when the subsuming classifier is more general, just as accurate, and covers all of the problem space of the classifier it subsumes.
In the eighth step, LCS adopts a highly elitist genetic algorithm (GA) which will select two parent classifiers based on fitness (survival of the fittest).  Parents are selected from [C] typically using tournament selection.  Some systems have applied roulette wheel selection or deterministic selection, and have differently selected parent rules from either [P] - panmictic selection, or from [M]).  Crossover and mutation operators are now applied to generate two new offspring rules.  At this point, both the parent and offspring rules are returned to [P].  The LCS genetic algorithm is highly elitist since each learning iteration, the vast majority of the population is preserved.  Rule discovery may alternatively be performed by some other method, such as an estimation of distribution algorithm, but a GA is by far the most common approach.  Evolutionary algorithms like the GA employ a stochastic search, which makes LCS a stochastic algorithm.  LCS seeks to cleverly explore the search space, but does not perform an exhaustive search of rule combinations, and is not guaranteed to converge on an optimal solution.
The last step in a generic LCS learning cycle is to maintain the maximum population size. The deletion mechanism will select classifiers for deletion (commonly using roulette wheel selection).  The probability of a classifier being selected for deletion is inversely proportional to its fitness.  When a classifier is selected for deletion, its numerosity parameter is reduced by one.  When the numerosity of a classifier is reduced to zero, it is removed entirely from the population.
LCS will cycle through these steps repeatedly for some user defined number of training iterations, or until some user defined termination criteria have been met.  For online learning, LCS will obtain a completely new training instance each iteration from the environment.  For offline learning, LCS will iterate through a finite training dataset.  Once it reaches the last instance in the dataset, it will go back to the first instance and cycle through the dataset again.
Once training is complete, the rule population will inevitably contain some poor, redundant and inexperienced rules.  It is common to apply a rule compaction, or condensation heuristic as a post-processing step.  This resulting compacted rule population is ready to be applied as a prediction model (e.g. make predictions on testing instances), and/or to be interpreted for knowledge discovery.
Whether or not rule compaction has been applied, the output of an LCS algorithm is a population of classifiers which can be applied to making predictions on previously unseen instances.  The prediction mechanism is not part of the supervised LCS learning cycle itself, however it would play an important role in a reinforcement learning LCS learning cycle.  For now we consider how the prediction mechanism can be applied for making predictions to test data.  When making predictions, the LCS learning components are deactivated so that the population does not continue to learn from incoming testing data.  A test instance is passed to [P] where a match set [M] is formed as usual.  At this point the match set is differently passed to a prediction array.  Rules in the match set can predict different actions, therefore a voting scheme is applied.  In a simple voting scheme, the action with the strongest supporting 'votes' from matching rules wins, and becomes the selected prediction.  All rules do not get an equal vote.  Rather the strength of the vote for a single rule is commonly proportional to its numerosity and fitness.  This voting scheme and the nature of how LCS's store knowledge, suggests that LCS algorithms are implicitly ensemble learners.
Individual LCS rules are typically human readable IF:THEN expression.  Rules that constitute the LCS prediction model can be ranked by different rule parameters and manually inspected.  Global strategies to guide knowledge discovery using statistical and graphical have also been proposed.[12][13]  With respect to other advanced machine learning approaches, such as artificial neural networks, random forests, or genetic programming, learning classifier systems are particularly well suited to problems that require interpretable solutions.
John Henry Holland was best known for his work popularizing genetic algorithms (GA), through his ground-breaking book "Adaptation in Natural and Artificial Systems"[14] in 1975 and his formalization of Holland's schema theorem.  In 1976, Holland conceptualized an extension of the GA concept to what he called a "cognitive system",[15] and provided the first detailed description of what would become known as the first learning classifier system in the paper "Cognitive Systems based on Adaptive Algorithms".[16]  This first system, named Cognitive System One (CS-1) was conceived as a modeling tool, designed to model a real system (i.e. environment) with unknown underlying dynamics using a population of human readable rules.  The goal was for a set of rules to perform online machine learning to adapt to the environment based on infrequent payoff/reward (i.e. reinforcement learning) and apply these rules to generate a behavior that matched the real system. This early, ambitious implementation was later regarded as overly complex, yielding inconsistent results.[2][17]
Beginning in 1980, Kenneth de Jong and his student Stephen Smith took a different approach to rule-based machine learning with (LS-1), where learning was viewed as an offline optimization process rather than an online adaptation process.[18][19][20]  This new approach was more similar to a standard genetic algorithm but evolved independent sets of rules.  Since that time LCS methods inspired by the online learning framework introduced by Holland at the University of Michigan have been referred to as Michigan-style LCS, and those inspired by Smith and De Jong at the University of Pittsburgh have been referred to as Pittsburgh-style LCS.[2][17]  In 1986, Holland developed what would be considered the standard Michigan-style LCS for the next decade.[21]
Other important concepts that emerged in the early days of LCS research included (1) the formalization of a bucket brigade algorithm (BBA) for credit assignment/learning,[22] (2) selection of parent rules from a common 'environmental niche' (i.e. the match set [M]) rather than from the whole population [P],[23] (3) covering, first introduced as a create operator,[24] (4) the formalization of an action set [A],[24] (5) a simplified algorithm architecture,[24] (6) strength-based fitness,[21] (7) consideration of single-step, or supervised learning problems[25] and the introduction of the correct set [C],[26] (8) accuracy-based fitness[27] (9) the combination of fuzzy logic with LCS[28] (which later spawned a lineage of fuzzy LCS algorithms), (10) encouraging long action chains and default hierarchies for improving performance on multi-step problems,[29][30][31] (11) examining latent learning (which later inspired a new branch of anticipatory classifier systems (ACS)[32]), and (12) the introduction of the first Q-learning-like credit assignment technique.[33]  While not all of these concepts are applied in modern LCS algorithms, each were landmarks in the development of the LCS paradigm.
Interest in learning classifier systems was reinvigorated in the mid 1990s largely due to two events; the development of the Q-Learning algorithm[34] for reinforcement learning, and the introduction of significantly simplified Michigan-style LCS architectures by Stewart Wilson.[11][35]  Wilson's Zeroth-level Classifier System (ZCS)[35] focused on increasing algorithmic understandability based on Hollands standard LCS implementation.[21]  This was done, in part, by removing rule-bidding and the internal message list, essential to the original BBA credit assignment, and replacing it with a hybrid BBA/Q-Learning strategy.  ZCS demonstrated that a much simpler LCS architecture could perform as well as the original, more complex implementations.  However, ZCS still suffered from performance drawbacks including the proliferation of over-general classifiers.
In 1995, Wilson published his landmark paper, "Classifier fitness based on accuracy" in which he introduced the classifier system XCS.[11]  XCS took the simplified architecture of ZCS and added an accuracy-based fitness, a niche GA (acting in the action set [A]), an explicit generalization mechanism called subsumption, and an adaptation of the Q-Learning credit assignment.  XCS was popularized by its ability to reach optimal performance while evolving accurate and maximally general classifiers as well as its impressive problem flexibility (able to perform both reinforcement learning and supervised learning). XCS later became the best known and most studied LCS algorithm and defined a new family of accuracy-based LCS.  ZCS alternatively became synonymous with strength-based LCS.  XCS is also important, because it successfully bridged the gap between LCS and the field of reinforcement learning.  Following the success of XCS, LCS were later described as reinforcement learning systems endowed with a generalization capability.[36] Reinforcement learning typically seeks to learn a value function that maps out a complete representation of the state/action space.  Similarly, the design of XCS drives it to form an all-inclusive and accurate representation of the problem space (i.e. a complete map) rather than focusing on high payoff niches in the environment (as was the case with strength-based LCS).  Conceptually, complete maps don't only capture what you should do, or what is correct, but also what you shouldn't do, or what's incorrect.  Differently, most strength-based LCSs, or exclusively supervised learning LCSs seek a rule set of efficient generalizations in the form of a best action map (or a partial map).   Comparisons between strength vs. accuracy-based fitness and complete vs. best action maps have since been examined in greater detail.[37][38]
XCS inspired the development of a whole new generation of LCS algorithms and applications.  In 1995, Congdon was the first to apply LCS to real-world epidemiological investigations of disease [39] followed closely by Holmes who developed the BOOLE++,[40] EpiCS,[41] and later EpiXCS[42] for epidemiological classification.  These early works inspired later interest in applying LCS algorithms to complex and large-scale data mining tasks epitomized by bioinformatics applications.  In 1998, Stolzmann introduced anticipatory classifier systems (ACS) which included rules in the form of 'condition-action-effect, rather than the classic 'condition-action' representation.[32]  ACS was designed to predict the perceptual consequences of an action in all possible situations in an environment.  In other words, the system evolves a model that specifies not only what to do in a given situation, but also provides information of what will happen after a specific action will be executed. This family of LCS algorithms is best suited to multi-step problems, planning, speeding up learning, or disambiguating perceptual aliasing (i.e. where the same observation is obtained in distinct states but requires different actions).  Butz later pursued this anticipatory family of LCS developing a number of improvements to the original method.[43]  In 2002, Wilson introduced XCSF, adding a computed action in order to perform function approximation.[44]  In 2003, Bernado-Mansilla introduced a sUpervised Classifier System (UCS), which specialized the XCS algorithm to the task of supervised learning, single-step problems, and forming a best action set.  UCS removed the reinforcement learning strategy in favor of a simple, accuracy-based rule fitness as well as the explore/exploit learning phases, characteristic of many reinforcement learners.  Bull introduced a simple accuracy-based LCS (YCS)[45] and a simple strength-based LCS Minimal Classifier System (MCS)[46] in order to develop a better theoretical understanding of the LCS framework.  Bacardit introduced GAssist[47] and BioHEL,[48] Pittsburgh-style LCSs designed for data mining and scalability to large datasets in bioinformatics applications.  In 2008, Drugowitsch published the book titled "Design and Analysis of Learning Classifier Systems" including some theoretical examination of LCS algorithms.[49]  Butz introduced the first rule online learning visualization within a GUI for XCSF[1] (see the image at the top of this page).  Urbanowicz extended the UCS framework and introduced ExSTraCS, explicitly designed for supervised learning in noisy problem domains (e.g. epidemiology and bioinformatics).[50]  ExSTraCS integrated (1) expert knowledge to drive covering and genetic algorithm towards important features in the data,[51] (2) a form of long-term memory referred to as attribute tracking,[52] allowing for more efficient learning and the characterization of heterogeneous data patterns, and (3) a flexible rule representation similar to Bacardit's mixed discrete-continuous attribute list representation.[53]  Both Bacardit and Urbanowicz explored statistical and visualization strategies to interpret LCS rules and perform knowledge discovery for data mining.[12][13]  Browne and Iqbal explored the concept of reusing building blocks in the form of code fragments and were the first to solve the 135-bit multiplexer benchmark problem by first learning useful building blocks from simpler multiplexer problems.[54] ExSTraCS 2.0 was later introduced to improve Michigan-style LCS scalability, successfully solving the 135-bit multiplexer benchmark problem for the first time directly.[5]  The n-bit multiplexer problem is highly epistatic and heterogeneous, making it a very challenging machine learning task.
Michigan-Style LCSs are characterized by a population of rules where the genetic algorithm operates at the level of individual rules and the solution is represented by the entire rule population.  Michigan style systems also learn incrementally which allows them to perform both reinforcement learning and supervised learning, as well as both online and offline learning.  Michigan-style systems have the advantage of being applicable to a greater number of problem domains, and the unique benefits of incremental learning.
Pittsburgh-Style LCSs are characterized by a population of variable length rule-sets where each rule-set is a potential solution.  The genetic algorithm typically operates at the level of an entire rule-set.  Pittsburgh-style systems can also uniquely evolve ordered rule lists, as well as employ a default rule.  These systems have the natural advantage of identifying smaller rule sets, making these systems more interpretable with regards to manual rule inspection.
Systems that seek to combine key strengths of both systems have also been proposed.
The name, "Learning Classifier System (LCS)", is a bit misleading since there are many machine learning algorithms that 'learn to classify' (e.g. decision trees, artificial neural networks), but are not LCSs.  The term 'rule-based machine learning (RBML)' is useful, as it more clearly captures the essential 'rule-based' component of these systems, but it also generalizes to methods that are not considered to be LCSs (e.g. association rule learning, or artificial immune systems). More general terms such as, 'genetics-based machine learning', and even 'genetic algorithm'[39] have also been applied to refer to what would be more characteristically defined as a learning classifier system.  Due to their similarity to genetic algorithms, Pittsburgh-style learning classifier systems are sometimes generically referred to as 'genetic algorithms'.  Beyond this, some LCS algorithms, or closely related methods, have been referred to as 'cognitive systems',[16] 'adaptive agents', 'production systems', or generically as a 'classifier system'.[55][56]   This variation in terminology contributes to some confusion in the field.
Up until the 2000s nearly all learning classifier system methods were developed with reinforcement learning problems in mind. As a result, the term ‘learning classifier system’ was commonly defined as the combination of ‘trial-and-error’ reinforcement learning with the global search of a genetic algorithm. Interest in supervised learning applications, and even unsupervised learning have since broadened the use and definition of this term.
q X3  Co-training is a machine learning algorithm used when there are only small amounts of labeled data and large amounts of unlabeled data. One of its uses is in text mining for search engines. It was introduced by Avrim Blum and Tom Mitchell in 1998.
Co-training is a semi-supervised learning technique that requires two views of the data. It assumes that each example is described using two different feature sets that provide different, complementary information about the instance. Ideally, the two views are conditionally independent (i.e., the two feature sets of each instance are conditionally independent given the class) and each view is sufficient (i.e., the class of an instance can be accurately predicted from each view alone). Co-training first learns a separate classifier for each view using any labeled examples. The most confident predictions of each classifier on the unlabeled data are then used to iteratively construct additional labeled training data.[1]
The original co-training paper described experiments using co-training to classify web pages into "academic course home page" or not; the classifier correctly categorized 95% of 788 web pages with only 12 labeled web pages as examples.[2] The paper has been cited over 1000 times, and received the 10 years Best Paper Award at the 25th International Conference on Machine Learning (ICML 2008), a renowned computer science conference.[3][4]
Krogel and Scheffer showed in 2004 that co-training is only beneficial if the data sets used in classification are independent. Co-training can only work if one of the classifiers correctly labels a piece of data that the other classifier previously misclassified. If both classifiers agree on all the unlabeled data, i.e. they are not independent, labeling the data does not create new information. When they applied co-training to problems in functional genomics, co-training worsened the results as the dependence of the classifiers was greater than 60%.[5]
Co-training has been used to classify web pages using the text on the page as one view and the anchor text of hyperlinks on other pages that point to the page as the other view. Simply put, the text in a hyperlink on one page can give information about the page it links to.[2] Co-training can work on "unlabeled" text that has not already been classified or tagged, which is typical for the text appearing on web pages and in emails. According to Tom Mitchell, "The features that describe a page are the words on the page and the links that point to that page. The co-training models utilize both classifiers to determine the likelihood that a page will contain data relevant to the search criteria." Text on websites can judge the relevance of link classifiers, hence the term "co-training". Mitchell claims that other search algorithms are 86% accurate, whereas co-training is 96% accurate.[6]
Co-training was used on FlipDog.com, a job search site, and by the U.S. Department of Labor, for a directory of continuing and distance education.[6] It has been used in many other applications, including statistical parsing and visual detection.[7]
q!X`  When we don’t know much about a new technology, we talk in generalisations. Those generalisations are often also extreme: the utopian drives of those who are developing it on one hand, and the dystopian visions that help society look before it leaps on the other. These tensions are true for machine learning, the set of techniques that enables much of what we currently think of as Artificial Intelligence. But, as the Royal Society’s recently published report Machine learning: the power and promise of computers that learn by example showed, we are already at the point where we can do better than the generalisations; give members of the public the opportunity to interrogate the “experts” and explore the future, and they come up with nuanced expectations in which context is everything. The Society’s report was informed by structured public dialogue, carried out over six days in four locations around the UK, with participants from mixed socio-economic backgrounds. Quantitative research showed only 9% of people have heard of machine learning, even though most of us use it regularly through applications such as text prediction or customer recommendation services. The public dialogue gave people the opportunity to discuss the science with leading academics. The conversations were seeded with near-term realistic scenarios from contexts such as GP’s surgeries and schools. The results showed common themes but they also revealed how, when it came to balancing potential risks and benefits, people gave very different weightings depending on what was at stake. Participants talked about potential advantages such as objectivity and accuracy: better an expert and well-tested diagnostic system than a human doctor unable to keep up with the latest literature and over-tired on the day. They raised the benefits of true efficiency in public services: systems that might relieve pressure on front line workers such as police officers or teachers. Even in time-limited discussions, participants often came up with ideas as to how machine learning could enhance rather than simply replace existing tasks or jobs. And they saw the potential for machine learning to address large-scale societal challenges such as climate change. At the same time, they were concerned about the depersonalisation of key services. The tired human doctor would still be essential to any conversation with the patient about the meaning of an important diagnosis (and some were sceptical about the likelihood of accurate diagnostic systems for mental illnesses, at least in the near term). They discussed whether the use of machine learning systems to augment experiences they currently enjoyed – from driving to writing poetry – might make these experiences less personal or ‘human’. They wanted to know the real limits of systems’ abilities to account for the full range of human characteristics. They wanted to avoid being stereotyped and having their choices of goods, services or news narrowed. They were also concerned about harm to individuals, with some suggestion that the physical embodiment of machine learning systems, as opposed to their use in classification or diagnosis, heightened the nature of those concerns. Context determined what people thought of the relative significance of the benefits and harms and the practical implications. In a qualitative assessment exercise, for example, they were highly positive about healthcare applications, considering these to be in the “high social value, low social risk” category, while assigning shopper recommendation systems (other than for financial services such as insurance and loans) and art as lower social value and higher social risk. When discussing the implications of potential physical harm from embodied systems such as autonomous vehicles or the social care robots in the home, they considered a range of potential levels of assurance. To ensure confidence they wanted evidence that the machine learning system was more accurate or safer than a human with an equivalent function. In high-stakes cases they saw an ongoing need for a “human in the loop”, either taking key decisions with the machine’s help, or in an oversight role. The potential implications of machine learning for jobs came up spontaneously and frequently. Participants were quick to make comparisons with previous industrial transitions such as the factory assembly line. Our dialogues only scratched the surface of this debate, which goes beyond technology and policy to impact business models, organisational structures, notions of ownership and rights. Other parts of the project highlighted different aspects of these transitions. A workshop with leaders in the professions began to explore current practices of continuous professional development and the changing nature of the ‘secular and sacred’ (technical and ethical) elements of professional practice. Workshops with different industrial sectors showed how difficult it is for smaller businesses to identify when and how they might be able to create value from machine learning, and how difficult it is to find the best advice in an area where demand for experts far outstrips supply. A workshop with leaders in the legal profession began to explore the extent to which machine learning might significantly disrupt business models. Chatbots are already carrying out basic legal advisory tasks such as dealing with parking fines; machine learning systems are able to replace junior staff in searching legal texts; and machine learning is now used by major firms to inform strategy by, for example, predicting counterparty reactions in major corporate cases. The Royal Society’s work in this area will continue. Meanwhile, a report on the governance of data and its uses with the British Academy will be published later in the summer.  Claire Craig is Director of Policy at the Royal Society. Jessica Montgomery is a Senior Policy Adviser at the Royal Societyq"X@  Amazon’s machine-learning specialists uncovered a big problem: their new recruiting engine did not like women. The team had been building computer programs since 2014 to review job applicants’ résumés, with the aim of mechanizing the search for top talent, five people familiar with the effort told Reuters. Automation has been key to Amazon’s e-commerce dominance, be it inside warehouses or driving pricing decisions. The company’s experimental hiring tool used artificial intelligence to give job candidates scores ranging from one to five stars – much as shoppers rate products on Amazon, some of the people said. “Everyone wanted this holy grail,” one of the people said. “They literally wanted it to be an engine where I’m going to give you 100 résumés, it will spit out the top five, and we’ll hire those.” But by 2015, the company realized its new system was not rating candidates for software developer jobs and other technical posts in a gender-neutral way. That is because Amazon’s computer models were trained to vet applicants by observing patterns in résumés submitted to the company over a 10-year period. Most came from men, a reflection of male dominance across the tech industry. In effect, Amazon’s system taught itself that male candidates were preferable. It penalized résumés that included the word “women’s”, as in “women’s chess club captain”. And it downgraded graduates of two all-women’s colleges, according to people familiar with the matter.  Amazon edited the programs to make them neutral to these particular terms. But that was no guarantee that the machines would not devise other ways of sorting candidates that could prove discriminatory, the people said. The Seattle company ultimately disbanded the team by the start of last year because executives lost hope for the project, according to the people, who spoke on condition of anonymity. Amazon’s recruiters looked at the recommendations generated by the tool when searching for new hires, but never relied solely on those rankings, they said. Amazon declined to comment on the recruiting engine or its challenges, but the company says it is committed to workplace diversity and equality. The company’s experiment, which Reuters is first to report, offers a case study in the limitations of machine learning. It also serves as a lesson to the growing list of large companies including Hilton Worldwide Holdings and Goldman Sachs that are looking to automate portions of the hiring process. Some 55% of US human resources managers said artificial intelligence, or AI, would be a regular part of their work within the next five years, according to a 2017 survey by talent software firm CareerBuilder. Amazon’s experiment began at a pivotal moment for the world’s largest online retailer. Machine learning was gaining traction in the technology world, thanks to a surge in low-cost computing power. And Amazon’s Human Resources department was about to embark on a hiring spree; since June 2015, the company’s global headcount has more than tripled to 575,700 workers, regulatory filings show. So it set up a team in Amazon’s Edinburgh engineering hub that grew to around a dozen people. Their goal was to develop AI that could rapidly crawl the web and spot candidates worth recruiting, the people familiar with the matter said. The group created 500 computer models focused on specific job functions and locations. They taught each to recognize some 50,000 terms that were found on past candidates’ résumés. The algorithms learned to assign little significance to skills that were common across IT applicants, such as the ability to write various computer codes, the people said. Instead, the technology favored candidates who described themselves using verbs more commonly found on male engineers’ resumes, such as “executed” and “captured”, one person said. Gender bias was not the only issue. Problems with the data that underpinned the models’ judgments meant that unqualified candidates were often recommended for all manner of jobs, the people said. With the technology returning results almost at random, Amazon shut down the project, they said. Other companies are forging ahead, underscoring the eagerness of employers to harness AI for hiring. Kevin Parker, chief executive of HireVue, a startup near Salt Lake City, said automation is helping companies look beyond the same recruiting networks upon which they have long relied. His firm analyzes candidates’ speech and facial expressions in video interviews to reduce reliance on résumés. “You weren’t going back to the same old places; you weren’t going back to just Ivy League schools,” Parker said. His company’s customers include Unilever PLC and Hilton. Goldman Sachs has created its own résumé analysis tool that tries to match candidates with the division where they would be the “best fit”, the company said. LinkedIn, the world’s largest professional network, has gone further. It offers employers algorithmic rankings of candidates based on their fit for job postings on its site. Still, John Jersin, vice-president of LinkedIn Talent Solutions, said the service is not a replacement for traditional recruiters. “I certainly would not trust any AI system today to make a hiring decision on its own,” he said. “The technology is just not ready yet.” Some activists say they are concerned about transparency in AI. The American Civil Liberties Union is currently challenging a law that allows criminal prosecution of researchers and journalists who test hiring websites’ algorithms for discrimination. “We are increasingly focusing on algorithmic fairness as an issue,” said Rachel Goodman, a staff attorney with the Racial Justice Program at the ACLU. Still, Goodman and other critics of AI acknowledged it could be exceedingly difficult to sue an employer over automated hiring; job candidates might never know it was being used. As for Amazon, the company managed to salvage some of what it learned from its failed AI experiment. It now uses a “much watered-down version” of the recruiting engine to help with some rudimentary chores, including culling duplicate candidate profiles from databases, one of the people familiar with the project said. Another said a new team in Edinburgh has been formed to give automated employment screening another try, this time with a focus on diversity.q#X�#  Britain's gambling industry is booming, and new figures show almost one in 20 iPhone owners use sports betting apps, but a UK technology firm is using behavioural data to combat the onset of gambling addiction. Featurespace, the corporate spin-out of a University of Cambridge engineering department project, is using machine learning techniques to identify people showing patterns indicative of problem gambling, before consulting psychologists on the best and safest preventative action to take. Gaming sites collect data on the betting patterns of every one of their customers, including the time of day, frequency and size of bets placed and the types of games an individual typically plays. By analysing this information, Featurespace is able to build up a picture of what is normal for any given individual, and what would constitute erratic or uncharacteristically risky behaviour that might indicate the onset of a gambling problem. Initially a consultancy, the company began working on fraud-detection solutions after winning a contract in 2008, and it was from its work in this area with gaming companies that the idea of looking at addictive behaviour emerged. "As we worked with the gaming industry, we know a lot of the companies really well, it was looking at one aspect of fraud - first party fraud - which is where a customer will charge back a transaction and say "well I didn't make that", though sometimes they actually did make that transaction but they may have been spending a lot more than they wanted to", said David Excell, co-founder and CTO. "So it's that false claim, maybe out of desperation, that constitutes the fraud, and we decided to start looking at the protections in place for customers at the moment to stop this kind of thing happening. We decided since we're harvesting so much data for our fraud solution work, how can we use some of that to try to understand the player from a corporate social responsibility point of view, to understand "is that player in control?" and so on", said Excell. When successful, there is not doubt that the results of Featurespace's work are providing a public service, but there are also commercial benefits for the gaming companies when addiction can be thwarted. "Commercially, the worst case scenario with a problem gambler is that the customer self-excludes - stops being a source of revenue altogether - so while we're effectively helping people stop developing a gambling addiction, a part of this is about serving the best interests of the gaming company. "If you can help that player have long term sustainable activity, then over the long term that customer will be of more value to you than if they make a short term loss, decide they are out of control and withdraw completely", said Excell. Gambling companies tend to be fairly footloose - they will often relocate depending on which country is the most lenient or has the most favourable tax rates at a particular time - take Gibraltar for example, host to dozens of gaming firms including Ladbrokes, Bwin and Victor Chandler (BetVictor). Leniency, in this case, concerns regulatory pressures. Gambling firms are more likely to set up in countries where they have more freedom to operate. "Part of the incentive to tackle addiction comes down to protecting business. A lot of countries will tighten regulations on gambling when it becomes a social issue, so as the number of problem gamblers rises, a firm may find conditions become less favourable. "Firms are trying to do everything they can to try and keep things at a point where legislators are not under public pressure to bring in more stringent laws", said Matt Mills, commercial director at Featurespace. "And there's also brand reputation for the operator. No company wants to be named in a case study of extreme gambling addiction, to be named in relation to a problem gambler losing their house", said Excell. In traditional fraud-detection systems, alarms are raised when certain thresholds are crossed, but with gaming addiction, this rule-based approach is all but ineffectual. "Where our technology really is works is that we get to know the habits of each individual. We start to learn an individual's playing patterns, and to what extent they are predictable. Our hypothesis around this is that where a player is in control of their gambling, their gaming patterns will be relatively regular. "That definition of 'regular' will be defined based on who they are, so if they play a lot, that's not necessarily a problem, but if we start seeing more erratic behaviour, rash decisions, playing at random times of the day, alarm bells will start ringing", said Excell. As well as patterns in the kinds of bets that are made and their timing, it is also important to account for customers' different financial constraints. "You could have a city trader and someone on minimum wage, and their ability to absorb different losses would be completely different. You can't just apply thresholds and rules across the spectrum", said Excell. Even the same betting pattern seen in two individuals could mean two completely different things, so a proper understanding of the context in which gamblers' decisions are made is vital. "We have data on such a diverse range of gaming patterns, and by their very nature we're also dealing with random outcomes - slot machines, roulette tables and the like - so working out what people are likely to do next gets very difficult. "By working with specialists we can look at how people increase their risk level or stake level to chase losses, and try to understand why people might be making the decisions we're seeing", said Excell. Having identified an individual as a problem gambler, the next step the gaming company takes is crucial. A wrong move, and the customer could simply head elsewhere to carry on where they left off. "We work with psychologists to try to establish how best to communicate with a customer identified as a potential problem gambler. People don't necessarily want to be told they have a problem - in many cases this can exacerbate the issue - so we need to make sure we're treading very carefully. "Most operators regulated by the UK market will offer session limits, deposit limits and loss limits, so what they might do is contact the individual and ask if they are aware of this functionality. The key is to be more suggestive than authoritative", said Excell. Another option available to the betting firms is to use advertising banners to direct customers to games that are more likely to ease them away from the site. "A lot of casino games operate around a return-to-player rate (RTP) whereby if the customer pays, say £100, the game would be set up to pay back an average of £90. Different games will have different RTPs, and there are a few schools of thought on whether certain rates have different impacts on somebody's likelihood of becoming addicted. "Some believe that if you lose really quickly, you'll be out of funds very quickly and will leave, and that a higher RTP will keep people on site, but others disagree", said Excell. An altogether different consideration for a gaming site that believes it has identified an at-risk individual is whether or not they are then deemed to have adopted a duty of care. "It's a really interesting one for the operators. Unless stipulated by a regulator, if they're aware that a customer may be a problem gambler, are they then liable for any loss? It's really difficult from a legal point of view for them to decide where they sit", said Excell. Another area where Featurespace is seeing worrying patterns emerge is in an entirely different sphere: social gaming. The likes of Farmville and Candy Crush - games played over social networks or as smartphone apps - may seem innocent enough, but there are risks, especially for young users. Although players are not obliged to part with any funds, upgrades or access to new levels can be bought for a nominal amount, but the costs an stack up. "Social gaming is not actually under the jurisdiction of gaming regulators, because you don't win money per se, but you're still being incentivised to deposit funds to buy upgrades. "We're seeing people becoming addicted in-game, but it's not actually gambling. There are considerable financial costs involved, but no regulation around what that actually means. Many games have no age restrictions, so you can have very young people paying their 69p to get to the next level", said Excell. "You only have to look at the App Store and see that the highest revenue earner is Candy Crush, which is making over $600,000 per day, entirely through people paying for additional in-game bonuses. This is a huge amount of money, and is completely unregulated", said Mills. Where successful in preventing addiction, Featurespace are tackling a social problem, but to what extent do you think this is just a happy coincidence brought about by trying to ensure a steady revenue stream for the gaming firms? Join the debate by leaving a comment below or contacting me directly on Twitter at @jburnmurdoch or @GuardianData.q$X�  Actor and equality campaigner Geena Davis has announced that Disney has adopted a digital tool that will analyse scripts and identify opportunities to rectify any gender and ethnic biases. Davis, founder of the Geena Davis Institute on Gender in Media, was speaking at the Power of Inclusion event in New Zealand, where she outlined the development of GD-IQ: Spellcheck for Bias, a machine learning tool described as “an intervention tool to infuse diversity and inclusion in entertainment and media”. Developed by the University of Southern California Viterbi School of Engineering, the Spellcheck for Bias is designed to analyse a script and determine the percentages of characters’ “gender, race, LGBTQIA [and] disabilities”. It can also track the percentage of “non-gender-defined speaking characters”. Davis said that Disney had partnered with her institute to pilot the project: “We’re going to collaborate with Disney over the next year using this tool to help their decision-making [and] identify opportunities to increase diversity and inclusion in the manuscripts that they receive. We’re very excited about the possibilities with this new technology and we encourage everybody to get in touch with us and give it a try.” Davis said the plan was not to “shame and blame”, but to reveal any unconscious biases in film projects before they entered production. The Geena Davis Institute on Gender in Media was founded in 2007 after the actor – who won a best supporting actress Oscar in 1989 for The Accidental Tourist – became concerned about the male-dominated entertainment her young daughter was consuming. At the summit, Davis added: “We don’t have enough female role models to inspire change. We need to see it in fiction to create the cultural change we need. If we see more women on screen as corporate leaders, scientists and politicians, there will be more women in real life taking up these roles.”q%X  A few times a year, the Guardian’s Digital Development department organises a so-called ‘hack day’. These events, which actually take place over two days, are a great chance to get out of the office, try new technologies and hack on fun stuff. We also invite our colleagues from outside the department to join us, so it’s an opportunity to increase our interaction with journalists and other interesting Guardian people.  This year’s summer hack day was held at the beautiful Shoreditch Town Hall and caffeinated by Noble Espresso. The first morning was dedicated to generating ideas, then we hacked through the afternoon and the next morning. Finally each team presented their hack, people voted on their favourites, and we had a party to celebrate some great hacks. Our hack was an experiment in helping our discussion moderators to find content that is likely to attract abusive or offensive comments. We receive tens of thousands of comments on the site every day, and our dedicated team of moderators works around the clock to find and remove any comments that do not meet our community standards. With so much new content being published every day, and an ever-increasing volume of comments, we wanted to help the moderators get on top of this workload and find the needles in the haystack. Our hypothesis was that the number of problematic comments on an article follows a pattern, i.e. it can be predicted using attributes such as the words in the article’s headline, the section of the site in which the article is found, the tags describing the article, and so on. We decided to try using Amazon Machine Learning (AML) to train a model and perform predictions using regression analysis.  As the feature to predict, we invented a metric that we called the “removed ratio”, i.e. the number of comments removed by moderators divided by the total number of comments on the article. Our thinking was that a discussion with a lot of comments that need to be removed is one that should be watched by moderators. We calculated the removed ratio for about 200,000 existing discussions, queried the Content API for some extra fields that we hypothesised to be relevant, and uploaded it all as a CSV file to AML. From there it was relatively simple to follow Amazon’s wizard to build, train and evaluate a model. When you import your CSV file into AML, a wizard guides you through the process of defining the schema, which basically involves specifying the data type of each field. It’s a good idea to include a header row in your CSV file so AML can give the fields sensible names. It also asks you to pick a target, i.e. the field that you are trying to predict. The type of this field decides what kind of model AML will build. In our case the removed ratio was a numeric field, so AML performed linear regression analysis to try to fit all the data points to a multi-dimensional line. One nice feature of AML is that it supports a free text type. You can give it data like an article headline and it will take care of tokenising and vectorising the text to extract features. This process is a black box, however, so it’s unclear whether it’s using techniques such as tf-idf to weight features. Once AML has built and trained the model, it can also evaluate it for you. It will automatically split your input data into training and evaluation data, using 70% of the records to train the model and the remaining 30% to evaluate it. After evaluation is complete you can visually explore the results, and AML will tell you how it compares with a baseline model that simply chooses the median value for all predictions. Rather depressingly, we struggled to get better performance than the baseline model. We tried building a number of models using different combinations of features, but it turns out that most of the features we chose (headline, tags, and so on) did not have much correlation to the feature we were trying to predict. In other words, we were asking AML to fit a straight line to a cloud of randomly scattered points. As the old data science proverb goes, “garbage in, garbage out”. Interestingly, the article headline was the most useful of all the features we tried, with a correlation of 0.23759. But just because our numbers were nonsense, that wouldn’t stop us from building a shiny demo! We wrote a simple Play app that grabbed the latest articles from the Content API, asked AML for a removed-ratio prediction for each one, and displayed them sorted by descending score. Although the numerical performance of the model is poor, the results it picks seem intuitively pretty good. The top few results include articles about Israel, climate change and gay marriage, all of which we would expect to attract some comments that require moderation. Trying out a few predictions, the model seemed to be reasonably good at predicting outliers, understanding that articles about fluffy kittens are unlikely to attract abusive comments while those about feminism tend to garner many more, but any predictions about more ‘normal’ articles were just stabs in the dark. In conclusion, Amazon Machine Learning is a really useful tool and we’re glad we got the chance to learn how to use it. While AWS user interfaces can be hit or miss, this one was really easy to use. It made the process of building, training and evaluating models much simpler and faster than if we were to do it ourselves using tools such as MLlib or SciPy. If we find a problem in the future that’s more appropriate for linear regression, we’d like to give AML another try. If you’d like to see the code we wrote, it’s available on GitHub.q&X*  Researchers working for Google have produced a new kind of computer intelligence which can learn in ways less immediately dependent on its programmers than any previous model. It can, for instance, navigate its way through a map of the London underground without being explicitly instructed how to do so. For the moment, this approach is less efficient than the old-fashioned, more specialised forms of artificial intelligence, but it holds out promise for the future and, like all such conceptual advances in computer programming, it raises more urgently the question of how society should harness these powers. Algorithms in themselves long predate computers. An algorithm is simply a sequence of instructions. Law codes can be seen as algorithms. The rules of games can be understood as algorithms, and nothing could be more human than making up games. Armies are perhaps the most completely algorithmic forms of social organisation. Yet too much contemporary discussion is framed as if the algorithmic workings of computer networks are something entirely new. It’s true that they can follow instructions at superhuman speed, with superhuman fidelity and over unimaginable quantities of data. But these instructions don’t come from nowhere. Although neural networks might be said to write their own programs, they do so towards goals set by humans, using data collected for human purposes. If the data is skewed, even by accident, the computers will amplify injustice. If the measures of success that the networks are trained against are themselves foolish or worse, the results will appear accordingly. Recent, horrifying examples include the use of algorithms to grade teachers in the US and to decide whether prisoners should be granted parole or not. In both these cases, the effect has been to punish the poor just for being poor. This kind of programming is, in the programmer Maciej Cegłowski’s phrase, like money-laundering for bias. But because the obnoxious determinations are made by computer programs, they seem to have an unassailable authority. We should not grant them this. Self-interest, as well as justice, is on the side of caution here. Algorithmic trading between giant banks is certainly to blame for such phenomena as “flash crashes” and is very plausibly responsible for the great financial disaster of 2008. But there is nothing inevitable about the decision to hand over to a computer the capacity to make decisions for us. There is always a human responsibility and this belongs with the companies or organisations that make use of – or at least unleash – the powers of the computer networks. To pretend otherwise is like blaming the outbreak of the first world war on railway timetables and their effect on the mobilisation of armies. The cure for the excesses of computerised algorithms is not in principle different from the remedies we have already discovered for algorithms that are embedded in purely human institutions. Expert claims must be scrutinised by outsiders and justified to sceptical, if intelligent and fair minded, observers. There needs to be a social mechanism for appealing against these judgments, and means to identify their mistakes and prevent them happening in future. The interests of the powerful must not be allowed to take precedence over the interests of justice and of society as a whole.q'X�  MALLET is a Java "Machine Learning for Language Toolkit".
MALLET is an integrated collection of Java code useful for statistical natural language processing, document classification, cluster analysis, information extraction, topic modeling and other machine learning applications to text.
MALLET was developed primarily by Andrew McCallum, of the University of Massachusetts Amherst, with assistance from graduate students and faculty from both UMASS and the University of Pennsylvania.

q(X  
Discriminative : The messenger RNA expression profile was the strongest predictor of metastasis, says Harpreet Kaur.
   Using the expression of 17 key genes (messenger RNAs) it is now possible to distinguish primary and metastatic cutaneous melanoma, which is the most common type of skin cancer. While 11 of the 17 genes have already been reported by other studies for cutaneous melanoma, it is for the first time that the potential role of remaining six genomic signatures in classifying samples as either primary or metastatic skin cutaneous cancer has been made. The 17 genomic signatures, which were identified by a team led by Prof. Gajendra P.S. Raghava from the Indraprastha Institute of Information Technology (IIIT), New Delhi, have high accuracy — over 89% — in discriminating metastatic from primary skin melanoma. These signatures also have high sensitivity (in case tumour is metastatic), and high specificity (in case the tumour is primary). The results were published in the journal Scientific Reports. Unlike in the case of primary skin melanoma, people with metastatic cutaneous melanoma have reduced survival rate and higher mortality rates. It therefore becomes important to be able to identify and classify skin cutaneous melanoma as either primary or metastatic so correct therapeutic strategies can be chalked out and survival rates improved in patients. Six machine learning models were used to study and validate the genomic signatures. They used expression profile of messenger RNA, micro RNA and methylation profile for discriminating tumour as primary or metastatic. “We found the messenger RNA expression profile was the strongest predictor of metastasis. The mRNA expression profile performed better than micro RNA and methylation profile of the patients,” says Harpreet Kaur from Institute of Microbial Technology (CSIR-IMTECH), Chandigarh and one of the first authors of the paper. “Of the six models used, one (SVC-W) model showed better ability to discriminate metastatic from primary tumours of validation dataset with overall accuracy of over 89%.” While messenger RNA outperformed microRNA in discriminating the status of the tumour, a particular microRNA was found to be a “strong predictor of metastatic melanoma”. Besides helping in distinguishing the kind of melanoma, the genomic signatures can also help in further categorising different stages of metastasis. For instance, it can tell if the tumour has spread to lymphatic nodes, which is an early stage of metastasis. Also, it can tell if the cancer has spread to distant parts of the body, which is a late stage of metastasis, says Dr. Sherry Bhalla from IIIT Delhi and the other first author. Six machine learning models were tested and used for classifying the tumour as either primary or metastatic. Of the six models, one model — Support Vector Classification with Weight (SVC-W) — has an accuracy of nearly 89.5%. The researchers have fiurther ntegrated the major prediction models in the webserver called CancerSPP that will help clinicians in classifying cutaneous melanoma as primary or metastatic using RNA sequence data, microRNA and methylation expression data. “It will also help in knowing the different states of metastatic samples,” says Kaur. “The analysis module in the CancerSPP webserver will provide information on the role of each of the important genes in various stages of metastasis and whether the expression of a gene is up-regulated or down-regulated.” You have reached your limit for free articles this month. Register to The Hindu for free and get unlimited access for 30 days.  Already have an account ? Sign in
 Sign up for a 30-day free trial. Sign Up Find mobile-friendly version of articles from the day's newspaper in one easy-to-read list. Enjoy reading as many articles as you wish without any limitations. A select list of articles that match your interests and tastes. Move smoothly between articles as our pages load instantly. A one-stop-shop for seeing the latest updates, and managing your preferences. We brief you on the latest and most important developments, three times a day. 
*Our Digital Subscription plans do not currently include the e-paper ,crossword, iPhone, iPad mobile applications and print. Our plans enhance your reading experience.
 
Why you should pay for quality journalism - Click to know more Please enter a valid email address. 
Printable version | Mar 2, 2020 11:47:37 PM | https://www.thehindu.com/sci-tech/science/now-machine-learning-based-model-can-determine-if-skin-cancer-has-spread/article29864091.ece
 
© THG PUBLISHING PVT LTD.
q)X�5  In May last year, a stunning report claimed that a computer program used by a US court for risk assessment was biased against black prisoners. The program, Correctional Offender Management Profiling for Alternative Sanctions (Compas), was much more prone to mistakenly label black defendants as likely to reoffend – wrongly flagging them at almost twice the rate as white people (45% to 24%), according to the investigative journalism organisation ProPublica. Compas and programs similar to it were in use in hundreds of courts across the US, potentially informing the decisions of judges and other officials. The message seemed clear: the US justice system, reviled for its racial bias, had turned to technology for help, only to find that the algorithms had a racial bias too. How could this have happened? The private company that supplies the software, Northpointe, disputed the conclusions of the report, but declined to reveal the inner workings of the program, which it considers commercially sensitive. The accusation gave frightening substance to a worry that has been brewing among activists and computer scientists for years and which the tech giants Google and Microsoft have recently taken steps to investigate: that as our computational tools have become more advanced, they have become more opaque. The data they rely on – arrest records, postcodes, social affiliations, income – can reflect, and further ingrain, human prejudice. The promise of machine learning and other programs that work with big data (often under the umbrella term “artificial intelligence” or AI) was that the more information we feed these sophisticated computer algorithms, the better they perform. Last year, according to global management consultant McKinsey, tech companies spent somewhere between $20bn and $30bn on AI, mostly in research and development. Investors are making a big bet that AI will sift through the vast amounts of information produced by our society and find patterns that will help us be more efficient, wealthier and happier. It has led to a decade-long AI arms race in which the UK government is offering six-figure salaries to computer scientists. They hope to use machine learning to, among other things, help unemployed people find jobs, predict the performance of pension funds and sort through revenue and customs casework. It has become a kind of received wisdom that these programs will touch every aspect of our lives. (“It’s impossible to know how widely adopted AI is now, but I do know we can’t go back,” one computer scientist says.) It’s impossible to know how widely adopted AI is now, but I do know we can’t go back But, while some of the most prominent voices in the industry are concerned with the far-off future apocalyptic potential of AI, there is less attention paid to the more immediate problem of how we prevent these programs from amplifying the inequalities of our past and affecting the most vulnerable members of our society. When the data we feed the machines reflects the history of our own unequal society, we are, in effect, asking the program to learn our own biases. “If you’re not careful, you risk automating the exact same biases these programs are supposed to eliminate,” says Kristian Lum, the lead statistician at the San Francisco-based, non-profit Human Rights Data Analysis Group (HRDAG). Last year, Lum and a co-author showed that PredPol, a program for police departments that predicts hotspots where future crime might occur, could potentially get stuck in a feedback loop of over-policing majority black and brown neighbourhoods. The program was “learning” from previous crime reports. For Samuel Sinyangwe, a justice activist and policy researcher, this kind of approach is “especially nefarious” because police can say: “We’re not being biased, we’re just doing what the math tells us.” And the public perception might be that the algorithms are impartial. We have already seen glimpses of what might be on the horizon. Programs developed by companies at the forefront of AI research have resulted in a string of errors that look uncannily like the darker biases of humanity: a Google image recognition program labelled the faces of several black people as gorillas; a LinkedIn advertising program showed a preference for male names in searches, and a Microsoft chatbot called Tay spent a day learning from Twitter and began spouting antisemitic messages. These small-scale incidents were all quickly fixed by the companies involved and have generally been written off as “gaffes”. But the Compas revelation and Lum’s study hint at a much bigger problem, demonstrating how programs could replicate the sort of large-scale systemic biases that people have spent decades campaigning to educate or legislate away. Computers don’t become biased on their own. They need to learn that from us. For years, the vanguard of computer science has been working on machine learning, often having programs learn in a similar way to humans – observing the world (or at least the world we show them) and identifying patterns. In 2012, Google researchers fed their computer “brain” millions of images from YouTube videos to see what it could recognise. It responded with blurry black-and-white outlines of human and cat faces. The program was never given a definition of a human face or a cat; it had observed and “learned” two of our favourite subjects. This sort of approach has allowed computers to perform tasks – such as language translation, recognising faces or recommending films in your Netflix queue – that just a decade ago would have been considered too complex to automate. But as the algorithms learn and adapt from their original coding, they become more opaque and less predictable. It can soon become difficult to understand exactly how the complex interaction of algorithms generated a problematic result. And, even if we could, private companies are disinclined to reveal the commercially sensitive inner workings of their algorithms (as was the case with Northpointe). Less difficult is predicting where problems can arise. Take Google’s face recognition program: cats are uncontroversial, but what if it was to learn what British and American people think a CEO looks like? The results would likely resemble the near-identical portraits of older white men that line any bank or corporate lobby. And the program wouldn’t be inaccurate: only 7% of FTSE CEOs are women. Even fewer, just 3%, have a BME background. When computers learn from us, they can learn our less appealing attributes. Joanna Bryson, a researcher at the University of Bath, studied a program designed to “learn” relationships between words. It trained on millions of pages of text from the internet and began clustering female names and pronouns with jobs such as “receptionist” and “nurse”. Bryson says she was astonished by how closely the results mirrored the real-world gender breakdown of those jobs in US government data, a nearly 90% correlation. “People expected AI to be unbiased; that’s just wrong. If the underlying data reflects stereotypes, or if you train AI from human culture, you will find these things,” Bryson says. People expected AI to be unbiased; that’s just wrong So who stands to lose out the most? Cathy O’Neil, the author of the book Weapons of Math Destruction about the dangerous consequences of outsourcing decisions to computers, says it’s generally the most vulnerable in society who are exposed to evaluation by automated systems. A rich person is unlikely to have their job application screened by a computer, or their loan request evaluated by anyone other than a bank executive. In the justice system, the thousands of defendants with no money for a lawyer or other counsel would be the most likely candidates for automated evaluation. In London, Hackney council has recently been working with a private company to apply AI to data, including government health and debt records, to help predict which families have children at risk of ending up in statutory care. Other councils have reportedly looked into similar programs. In her 2016 paper, HRDAG’s Kristian Lum demonstrated who would be affected if a program designed to increase the efficiency of policing was let loose on biased data. Lum and her co-author took PredPol – the program that suggests the likely location of future crimes based on recent crime and arrest statistics – and fed it historical drug-crime data from the city of Oakland’s police department. PredPol showed a daily map of likely “crime hotspots” that police could deploy to, based on information about where police had previously made arrests. The program was suggesting majority black neighbourhoods at about twice the rate of white ones, despite the fact that when the statisticians modelled the city’s likely overall drug use, based on national statistics, it was much more evenly distributed. As if that wasn’t bad enough, the researchers also simulated what would happen if police had acted directly on PredPol’s hotspots every day and increased their arrests accordingly: the program entered a feedback loop, predicting more and more crime in the neighbourhoods that police visited most. That caused still more police to be sent in. It was a virtual mirror of the real-world criticisms of initiatives such as New York City’s controversial “stop-and-frisk” policy. By over-targeting residents with a particular characteristic, police arrested them at an inflated rate, which then justified further policing. PredPol’s co-developer, Prof Jeff Brantingham, acknowledged the concerns when asked by the Washington Post. He claimed that – to combat bias – drug arrests and other offences that rely on the discretion of officers were not used with the software because they are often more heavily enforced in poor and minority communities. And while most of us don’t understand the complex code within programs such as PredPol, Hamid Khan, an organiser with Stop LAPD Spying Coalition, a community group addressing police surveillance in Los Angeles, says that people do recognise predictive policing as “another top-down approach where policing remains the same: pathologising whole communities”. There is a saying in computer science, something close to an informal law: garbage in, garbage out. It means that programs are not magic. If you give them flawed information, they won’t fix the flaws, they just process the information. Khan has his own truism: “It’s racism in, racism out.” It’s unclear how existing laws to protect against discrimination and to regulate algorithmic decision-making apply in this new landscape. Often the technology moves faster than governments can address its effects. In 2016, the Cornell University professor and former Microsoft researcher Solon Barocas claimed that current laws “largely fail to address discrimination” when it comes to big data and machine learning. Barocas says that many traditional players in civil rights, including the American Civil Liberties Union (ACLU), are taking the issue on in areas such as housing or hiring practices. Sinyangwe recently worked with the ACLU to try to pass city-level policies requiring police to disclose any technology they adopt, including AI. But the process is complicated by the fact that public institutions adopt technology sold by private companies, whose inner workings may not be transparent. “We don’t want to deputise these companies to regulate themselves,” says Barocas. In the UK, there are some existing protections. Government services and companies must disclose if a decision has been entirely outsourced to a computer, and, if so, that decision can be challenged. But Sandra Wachter, a law scholar at the Alan Turing Institute at Oxford University, says that the existing laws don’t map perfectly to the way technology has advanced. There are a variety of loopholes that could allow the undisclosed use of algorithms. She has called for a “right to explanation”, which would require a full disclosure as well as a higher degree of transparency for any use of these programs. The scientific literature on the topic now reflects a debate on the nature of “fairness” itself, and researchers are working on everything from ways to strip “unfair” classifiers from decades of historical data, to modifying algorithms to skirt round any groups protected by existing anti-discrimination laws. One researcher at the Turing Institute told me the problem was so difficult because “changing the variables can introduce new bias, and sometimes we’re not even sure how bias affects the data, or even where it is”. The institute has developed a program that tests a series of counterfactual propositions to track what affects algorithmic decisions: would the result be the same if the person was white, or older, or lived elsewhere? But there are some who consider it an impossible task to integrate the various definitions of fairness adopted by society and computer scientists, and still retain a functional program. “In many ways, we’re seeing a response to the naive optimism of the earlier days,” Barocas says. “Just two or three years ago you had articles credulously claiming: ‘Isn’t this great? These things are going to eliminate bias from hiring decisions and everything else.’” Meanwhile, computer scientists face an unfamiliar challenge: their work necessarily looks to the future, but in embracing machines that learn, they find themselves tied to our age-old problems of the past. Follow the Guardian’s Inequality Project on Twitter here, or email us at inequality.project@theguardian.comq*X�  Budding authors face a minefield when it comes to publishing their work. For a large fee, as much as $3,000, they can make their work available to anyone who wants to read it. Or they can avoid the fee and have readers pay the publisher instead. Often it is libraries that foot this bill through expensive annual subscriptions. This is not the lot of wannabe fiction writers, it’s the business of academic publishing. More than 200 years ago, Giuseppe Piazzi, an isolated astronomer in Palermo, Sicily, discovered a dwarf planet. For him, publishing meant writing a letter to his friend Franz von Zach. Each month von Zach collated letters from astronomers across Europe and redistributed them. No internet for these guys: they found out about the latest discoveries from leatherbound volumes of letters called Monatliche Correspondenz. The time it took to disseminate research threw up its own problems: by the time Piazzi’s data were published, the planet had vanished in the sun’s glare. It was a 23-year-old reader in Göttingen who saved the day. Using Kepler’s laws of planetary motion, Carl Friedrich Gauss calculated the location of what we know today as Ceres. Gauss, who became Germany’s greatest mathematician, and Piazzi shared their learnings freely, but they accepted the need to pay for the work that von Zach undertook. This is the closed-access publishing model. In my own field of machine learning, itself an academic descendant of Gauss’s pioneering work, modern data are no longer just planetary observations but medical images, spoken language, internet documents and more. The results are medical diagnoses, recommender systems, and whether driverless cars see stop signs or not. Machine learning is the field that underpins the current revolution in artificial intelligence. The ability to pay no longer determines the ability to play Machine learning is a young and technologically astute field. It does not have the historical traditions of other fields and its academics have seen no need for the closed-access publishing model. The community itself created, collated, and reviewed the research it carried out. We used the internet to create new journals that were freely available and made no charge to authors. The era of subscriptions and leatherbound volumes seemed to be behind us. The public already pays taxes that fund our research. Why should people have to pay again to read the results? Colleagues in less well-funded universities also benefit. Makerere University in Kampala, Uganda, has as much access to the leading machine-learning research as Harvard or MIT. The ability to pay no longer determines the ability to play. Machine learning has demonstrated that an academic field can not only survive, but thrive, without the involvement of commercial publishers. But this has not stopped traditional publishers from entering the market. Our success has caught their attention. Most recently, the publishing conglomerate Springer Nature announced a new journal targeted at the community called Nature Machine Intelligence. The publisher now has 53 journals that bear the Nature name. Should we be concerned? What would drive authors and readers towards a for-profit subscription journal when we already have an open model for sharing our ideas? Academic publishers have one card left to play: their brand. The diversity and quantity of academic research means that it is difficult for a researcher in one field to rate the work in another. Sometimes a journal’s brand is used as a proxy for quality. When academics look for promotion, having papers in a “brand-name journal” can be a big help. Nature is the Rolex of academic publishing. But in contrast to Rolex, whose staff are responsible for the innovation in its watches, Nature relies on academics to provide its content. We are the watchmakers, they are merely the distributors. Many in our research community see the Nature brand as a poor proxy for academic quality. We resist the intrusion of for-profit publishing into our field. As a result, at the time of writing, more than 3,000 researchers, including many leading names in the field from both industry and academia, have signed a statement refusing to submit, review or edit for this new journal. We see no role for closed access or author-fee publication in the future of machine-learning research. We believe the adoption of this new journal as an outlet of record for the machine-learning community would be a retrograde step.  Neil Lawrence is on leave of absence from the University of Sheffield and is working at Amazon. He is the founding editor of the freely available journal Proceedings of Machine Learning Research, which has to date published nearly 4,000 papers. The ideas in this article represent his personal opinion.q+XC  The excitement around artificial intelligence (AI) has created a dynamic where perception and reality are at odds: everyone assumes that everyone else is already using it, yet relatively few people have personal experience with it, and it’s almost certain that no one is using it very well. This is AI’s third cycle in a long history of hype – the first conference on AI took place 60 years ago this year – but what is better described as “machine learning” is still very young when it comes to how organisations implement it. While we all encounter machine learning whenever we use autocorrect, Siri, Spotify and Google, the vast majority of businesses are yet to grasp its promise, particularly when it comes to practically adding value in supporting internal decision making. Over the last few months I’ve been asking a wide range of leaders of large and small companies how and why they are using machine learning within their organisations. By exposing the areas of confusion, concerns and different approaches business leaders are taking, these conversations highlight five interesting lessons.  Far more important than the machine learning approach you take is the question you ask. Machine learning is not yet anywhere near “artificial general intelligence” – it remains a set of specialised tools, not a panacea.  For Deep Knowledge Ventures, the Hong Kong-based venture firm that added a machine learning algorithm named VITAL to its board in 2014, it was about adding a tool to analyse market data around investment opportunities. For global professional service firms experimenting in this space, machine learning could allow deeper and faster document analysis. Energy companies want to make better use of production and transport data to make resourcing decisions while one defence contractor is looking for “wiser” analysis of stakeholder networks in conflict zones. While there is widespread fear that AI will be used to automate in ways that creates mass employment, the vast majority of firms I spoke to are, at least at this stage, experimenting with machine learning to augment rather than replace human decision making. It’s therefore important to identify which processes and decisions could benefit from augmentation: is it about better contextual awareness or more efficient interrogation of proprietary data? Precise questions lead more easily to useful experimentation.  Machine learning relies on data – whether big or small. If your decisions revolve around deeper or faster analysis of your own data, it’s likely you’ll need to get that in order before you can do anything else. This could mean not just new databases and better data “hygiene”, but new inputs, new workflows and new information ontologies, all before you start to build the model that can take you towards recommendation or prediction to support decision making. Don’t forget to double down on your cyber security strategy if data is now flowing to and from new places. Data scientists are not cheap. Glassdoor lists the average salary of a data scientist in Palo Alto, California, as $130,000 (£100,000). And though you may not think you are competing with Silicon Valley salaries for talent, you are if you want great people: a great data scientist can easily be 50 times more valuable than a competent one, which means that both hiring and retaining them can be pricey.  You may opt to outsource many aspects of your machine learning, however, every company I spoke to, regardless of approach, said that machine learning had required a significant investment in their staff in terms of expanding both knowledge and skills.  The latest rage is bots – application programming interfaces (APIs) that use machine learning to do specialised tasks such as process speech, assess text for sentiment or tag concepts. Bots can be seen as a small and, but imperfect, part of “Machine learning as a service”. If the creator of Siri is right, there will be an entire ecosystem of machine learning APIs that write their own code to meet your needs. Companies like Salesforce have also started to integrate machine learning into their platforms, lowering the cost and friction of getting started. As the machine learning ecosystem evolves, companies will find interesting ways to combine in-house industry experience with a range of off-the-shelf tools and open source algorithms to create highly-customised decision-support tools.  Technologies are not “values-free” – all the tools we design, including AI systems, have a series of values, biases and assumptions built into them by their creators and reflected by the data they interrogate. Systems that use machine learning to make decisions for us can reflect or reinforce gender, racial and social biases. Compounding this, the perceived complexity of machine learning means that when it fails there is little recognition of harm and no appeal for those affected, thanks to what Cathy O’Neil calls “the authority of the inscrutable”. As we discussed during UCL School of Management debate on AI on Tuesday night, human beings need to be firmly at the centre of all our technological systems.  When our decisions are assisted by machine learning, the reasoning should be as transparent and verifiable as possible. For humans and intelligent machines to have a satisfying partnership, we need to ensure we learn from machines as much as they learn from us.  Nicholas Davis is head of society and innovation and a member of the executive committee of the World Economic Forum. He leads the organisation’s work on innovation, entrepreneurship and “the Fourth Industrial Revolution”. To get weekly news analysis, job alerts and event notifications direct to your inbox, sign up free for Media & Tech Network membership. All Guardian Media & Tech Network content is editorially independent except for pieces labelled “Paid for by” – find out more here.q,X3  The mean absolute percentage error (MAPE), also known as mean absolute percentage deviation (MAPD), is a measure of prediction accuracy of a forecasting method in statistics, for example in trend estimation, also used as a loss function for regression problems in machine learning. It usually expresses the accuracy as a ratio defined by the formula:
where At is the actual value and Ft is the forecast value. Note that the MAPE is also sometimes reported as a percentage, which is the above equation multiplied by 100. The difference between At and Ft is divided by the actual value At again.  The absolute value in this calculation is summed for every forecasted point in time and divided by the number of fitted points n. Multiplying by 100% makes it a percentage error.
Mean absolute percentage error is commonly used as a loss function for regression problems and in model evaluation, because of its very intuitive interpretation in terms of relative error.
Definition
Consider a standard regression setting in which the data are fully described by a random pair 



Z
=
(
X
,
Y
)


{\displaystyle Z=(X,Y)}

 with values in 





R


d


×

R



{\displaystyle \mathbb {R} ^{d}\times \mathbb {R} }

, and n i.i.d. copies 



(

X

1


,

Y

1


)
,
.
.
.
,
(

X

n


,

Y

n


)


{\displaystyle (X_{1},Y_{1}),...,(X_{n},Y_{n})}

 of 



(
X
,
Y
)


{\displaystyle (X,Y)}

. Regression models aims at finding a good model for the pair, that is a measurable function 



g


{\displaystyle g}

 from 





R


d




{\displaystyle \mathbb {R} ^{d}}

 to 




R



{\displaystyle \mathbb {R} }

 such that 



g
(
X
)


{\displaystyle g(X)}

 is “close to” 



Y


{\displaystyle Y}

 . 
In the classical regression setting, the closeness of 



g
(
X
)


{\displaystyle g(X)}

 to 



Y


{\displaystyle Y}

 is measured via the L2 risk, also called the Mean squared error (MSE). In the MAPE regression context[1], the closeness of 



g
(
X
)


{\displaystyle g(X)}

 to 



Y


{\displaystyle Y}

 is measured via the MAPE, and the aim of MAPE regressions is to find a model 




g

M
A
P
E




{\displaystyle g_{MAPE}}

 such that:





g

M
A
P
E


(
x
)
=
arg
⁡

min

g
∈


G





E


[


|



g
(
X
)
−
Y

Y


|


|

X
=
x

]



{\displaystyle g_{MAPE}(x)=\arg \min _{g\in {\mathcal {G}}}\mathbb {E} \left[\left|{\frac {g(X)-Y}{Y}}\right||X=x\right]}


where 





G




{\displaystyle {\mathcal {G}}}

 is the class of models considered (e.g. linear models).
In practice
In practice 




g

M
A
P
E


(
x
)


{\displaystyle g_{MAPE}(x)}

 can be estimated by the Empirical Risk Minimization strategy, leading to








g
^




M
A
P
E


(
x
)
=
arg
⁡

min

g
∈


G





∑

i
=
1


n



|



g
(

X

i


)
−

Y

i




Y

i




|



{\displaystyle {\widehat {g}}_{MAPE}(x)=\arg \min _{g\in {\mathcal {G}}}\sum _{i=1}^{n}\left|{\frac {g(X_{i})-Y_{i}}{Y_{i}}}\right|}


From a practical point of view, the use of the MAPE as a quality function for regression model is equivalent to doing weighted Mean absolute error (MAE) regression, also known as quantile regression. This property is trivial since








g
^




M
A
P
E


(
x
)
=
arg
⁡

min

g
∈


G





∑

i
=
1


n


ω
(

Y

i


)

|

g
(

X

i


)
−

Y

i



|



 with 


ω
(

Y

i


)
=

|


1

Y

i




|



{\displaystyle {\widehat {g}}_{MAPE}(x)=\arg \min _{g\in {\mathcal {G}}}\sum _{i=1}^{n}\omega (Y_{i})\left|g(X_{i})-Y_{i}\right|{\mbox{ with }}\omega (Y_{i})=\left|{\frac {1}{Y_{i}}}\right|}


As a consequence, the use of the MAPE is very easy in practice, for example using existing libraries for quantile regression allowing weights.
Consistency
The use of the MAPE as a loss function for Regression analysis is feasible both on a practical point of view and on a theoretical one, since the existence of an optimal model and the consistency of the Empirical risk minimization can be proved [1].
Problems can occur when calculating the MAPE value with a series of small denominators. A singularity problem of the form 'one divided by zero' and/or the creation of very large changes in the Absolute Percentage Error, caused by a small deviation in error, can occur.
As an alternative, each actual value (At) of the series in the original formula can be replaced by the average of all actual values (Āt) of that series. This alternative is still being used for measuring the performance of models that forecast spot electricity prices.[2]
Note that this is the same as dividing the sum of absolute differences by the sum of actual values, and is sometimes referred to as WAPE (weighted absolute percentage error).
Although the concept of MAPE sounds very simple and convincing, it has major drawbacks in practical application [3], and there are many studies on shortcomings and misleading results from MAPE.[4][5]
To overcome these issues with MAPE, there are some other measures proposed in literature: 
q-X7  Machine learning and artificial intelligence have the potential to make significant improvements to our lives in areas such as health and public services. However, as Ian Sample points out (Computer says no: why making AIs fair, open and accountable is crucial, 6 November), there are real concerns about fairness and accountability. The Royal Society and the British Academy, in Data Management and Use: Governance in the 21st century, make the urgent case for a stewardship body for data use. The governance response must be driven by the overarching principle of human flourishing – recognising that humans do not serve data, but that data must be used to serve humans and human communities. A number of principles follow from this, including the need to protect individual and collective rights and interests. We need an independent, interdisciplinary stewardship body that can identify where there are governance gaps, with the power to urge the right bodies to fill those gaps. Swift action is needed to ensure that this important area of technology operates in a way that deserves and secures public trust.Professor Dame Ottoline LeyserChair of the Royal Society Science Policy Advisory Group • Join the debate – email guardian.letters@theguardian.com • Read more Guardian letters – click here to visit gu.com/lettersq.Xj  In Euclidean geometry, linear separability is a property of two sets of points. This is most easily visualized in two dimensions (the Euclidean plane) by thinking of one set of points as being colored blue and the other set of points as being colored red. These two sets are linearly separable if there exists at least one line in the plane with all of the blue points on one side of the line and all the red points on the other side. This idea immediately generalizes to higher-dimensional Euclidean spaces if line is replaced by hyperplane.
The problem of determining if a pair of sets is linearly separable and finding a separating hyperplane if they are, arises in several areas.  In statistics and machine learning, classifying certain types of data is a problem for which good algorithms exist that are based on this concept.
Let 




X

0




{\displaystyle X_{0}}

 and 




X

1




{\displaystyle X_{1}}

 be two sets of points in an n-dimensional Euclidean space. Then 




X

0




{\displaystyle X_{0}}

 and 




X

1




{\displaystyle X_{1}}

 are linearly separable if there exist n + 1 real numbers 




w

1


,

w

2


,
.
.
,

w

n


,
k


{\displaystyle w_{1},w_{2},..,w_{n},k}

, such that every point 



x
∈

X

0




{\displaystyle x\in X_{0}}

 satisfies 




∑

i
=
1


n



w

i



x

i


>
k


{\displaystyle \sum _{i=1}^{n}w_{i}x_{i}>k}

 and every point 



x
∈

X

1




{\displaystyle x\in X_{1}}

 satisfies 




∑

i
=
1


n



w

i



x

i


<
k


{\displaystyle \sum _{i=1}^{n}w_{i}x_{i}<k}

, where 




x

i




{\displaystyle x_{i}}

 is the 



i


{\displaystyle i}

-th component of 



x


{\displaystyle x}

.
Equivalently, two sets are linearly separable precisely when their respective convex hulls are disjoint (colloquially, do not overlap).[citation needed]
Three non-collinear points in two classes ('+' and '-') are always linearly separable in two dimensions. This is illustrated by the three examples in the following figure (the all '+' case is not shown, but is similar to the all '-' case):
However, not all sets of four points, no three collinear, are linearly separable in two dimensions. The following example would need two straight lines and thus is not linearly separable:
Notice that three points which are collinear and of the form "+ ⋅⋅⋅ — ⋅⋅⋅ +" are also not linearly separable.
A Boolean function in n variables can be thought of as an assignment of 0 or 1 to each vertex of a Boolean hypercube in n dimensions. This gives a natural division of the vertices into two sets. The Boolean function is said to be linearly separable provided these two sets of points are linearly separable. The number of distinct Boolean functions is 




2


2

n






{\displaystyle 2^{2^{n}}}

where n is the number of variables passed into the function.[1]
Classifying data is a common task in machine learning.
Suppose some data points, each belonging to one of two sets, are given and we wish to create a model that will decide which set a new data point will be in. In the case of support vector machines, a data point is viewed as a p-dimensional vector (a list of p numbers), and we want to know whether we can separate such points with a (p − 1)-dimensional hyperplane. This is called a linear classifier. There are many hyperplanes that might classify (separate) the data. One reasonable choice as the best hyperplane is the one that represents the largest separation, or margin, between the two sets. So we choose the hyperplane so that the distance from it to the nearest data point on each side is maximized. If such a hyperplane exists, it is known as the maximum-margin hyperplane and the linear classifier it defines is known as a maximum margin classifier.
More formally, given some training data 





D




{\displaystyle {\mathcal {D}}}

, a set of n points of the form
where the yi is either 1 or −1, indicating the set to which the point 





x


i




{\displaystyle \mathbf {x} _{i}}

 belongs. Each 





x


i




{\displaystyle \mathbf {x} _{i}}

 is a p-dimensional real vector. We want to find the maximum-margin hyperplane that divides the points having 




y

i


=
1


{\displaystyle y_{i}=1}

 from those having 




y

i


=
−
1


{\displaystyle y_{i}=-1}

. Any hyperplane can be written as the set of points 




x



{\displaystyle \mathbf {x} }

 satisfying
where 



⋅


{\displaystyle \cdot }

 denotes the dot product and 





w




{\displaystyle {\mathbf {w} }}

 the (not necessarily normalized) normal vector to the hyperplane. The parameter 






b

‖

w

‖






{\displaystyle {\tfrac {b}{\|\mathbf {w} \|}}}

 determines the offset of the hyperplane from the origin along the normal vector 





w




{\displaystyle {\mathbf {w} }}

.
If the training data are linearly separable, we can select two hyperplanes in such a way that they separate the data and there are no points between them, and then try to maximize their distance.
q/X�#  A killer AI has gone on a rampage through Pakistan, slaughtering perhaps thousands of people. At least that’s the impression you’d get if you read this report from Ars Technica (based on NSA documents leaked by The Intercept), which claims that a machine learning algorithm guiding U.S. drones – unfortunately named ‘SKYNET’ – could have wrongly targeted numerous innocent civilians.  Let’s start with the facts. For the last decade or so, the United States has used unmanned drones to attack militants in Pakistan. The number of kills is unknown, but estimates start at over a thousand and range up to maybe four thousand. A key problem for the intelligence services is finding the right people to kill, since the militants are mixed in with the general population and not just sitting in camp together waiting to be bombed. One thing they have is data, which apparently includes metadata from 55 million mobile phone users in Pakistan. For each user they could see which cell towers were pinged, how they moved, who they called, who called them, how long they spent on calls, when phones were switched off, and any of several dozen other statistics. That opened up a possible route for machine learning, neatly summarised on slide 2 of this deck. If we know that some of these 55 million people are couriers, can an algorithm find patterns in their behaviour and spot others who act in a similar way? What exactly is a ‘courier’ anyway? This is important to understanding some of the errors that The Intercept and Ars Technica made. Courier isn’t a synonym for ‘terrorist’ as such - it means a specific kind of agent. Terrorist groups are justifiably nervous about using digital communications, and so a lot of messages are still delivered by hand, by couriers. Bin Laden made extensive use of couriers to pass information around, and it was through one of them – Abu Ahmed al-Kuwaiti (an alias) - that he was eventually found. That’s who the AI was being trained to detect – not the bin Ladens but the al-Kuwaitis. Not the targets so much as the people who might lead agents to them. Ars Technica implies that somehow the output of this courier detection method was used directly to “generate the final kill list” for drone strikes, but there’s zero evidence I can see that this was ever the case, and it would make almost no sense given what the algorithm was actually looking for - you don’t blow up your leads.  How did it work? The NSA tried several classification algorithms, and chose what’s known as a random forest approach. It’s actually pretty simple to describe. You have 55 million records, each with 80 different variables or ‘features’ in them. A random forest algorithm splits this data up into lots of random overlapping bundles of records and features. So you might end up with e.g.: And so on. The next step is to train a decision tree on each bundle of data. A decision tree is, very crudely speaking, an algorithm that takes a record with a number of variables and goes through a series of yes/no questions to reach an answer. So for example, ‘if this variable is > x1 and that variable is not > x2 and ‘a third variable’ is > x3…’ (...and so on for perhaps dozens of steps...) ‘…then this record is a courier.’ The exact values for all the ‘x’s used are learned by training the algorithm on some test data where the outcomes are known, and you can think of them collectively as a model of the real world.  Having created all those trees, you then bring them together to create your metaphorical forest. You run every single tree on each record, and combine the results from all of them to get some probability that the record is a courier. Very broadly speaking, the more the trees agree, the higher the probability is. Obviously this is a really simplified explanation, but hopefully it’s enough to show that we’re not talking about a mysterious black box here.  How well did the algorithm do? Both The Intercept and Ars Technica leapt on the fact that the person with the highest probability of being a courier that they found in the data was Ahmad Zaidan, a bureau chief for Al-Jazeera in Islamabad. Cue snorts of derision from Ars Technica: “As The Intercept reported, Zaidan frequently travels to regions with known terrorist activity in order to interview insurgents and report the news. But rather than questioning the machine learning that produced such a bizarre result, the NSA engineers behind the algorithm instead trumpeted Zaidan as an example of a SKYNET success in their in-house presentation, including a slide that labelled Zaidan as a ‘MEMBER OF AL-QA’IDA.’” If you knew nothing about machine learning, or you ignored the goals the algorithm was actually set, it might seem like a bad result. Actually it isn’t. Let’s ignore the NSA’s prior beliefs about the man. The algorithm was trained to look for ‘couriers’, people who carry messages to and from Al Qaida members. As a journalist, Zaidan was so well connected with Al Qaida members that he interviewed Bin Laden on at least two occasions. This was a man who regularly travelled to, spoke with and carried messages from Al Qaida members.  If the purpose of the algorithm had been narrowly to ‘detect terrorists’ or ‘identify suicide bombers’ then The Intercept might have a point. But it wasn’t. It was trained to find people tightly linked to Al Qaida who might be carrying useful intelligence. Its identification of Zaidan – regardless of whether he was acting as a journalist or not – was entirely correct within the context of those goals.  (As an aside, obviously I’m not making any moral statement here about the validity of intelligence agencies tracking journalists and intercepting their communications. I’m talking simply about the performance of the algorithm in carrying out the objectives it was set.)  So the one case that The Intercept and Ars Technica highlight as a failure of the algorithm is actually a pretty striking success story. Zaidan is exactly the kind of person the NSA would expect and want the algorithm to highlight. Of course it’s just one example thought, so how well did the algorithm perform over the rest of the data? The answer is: actually pretty well. The challenge here is pretty enormous because while the NSA has data on millions of people, only a tiny handful of them are confirmed couriers. With so little information, it’s pretty hard to create a balanced set of data to train an algorithm on – an AI could just classify everyone as innocent and still claim to be over 99.99% accurate. A machine learning algorithm’s basic job is to build a model of the world it sees, and when you have so few examples to learn from it can be a very cloudy view.  In the end though they were able to train a model with a false positive rate – the number of people wrongly classed as terrorists - of just 0.008%. That’s a pretty good achievement, but given the size of Pakistan’s population it still means about 15,000 people being wrongly classified as couriers. If you were basing a kill list on that, it would be pretty bloody awful. Here’s where The Intercept and Ars Technica really go off the deep end. The last slide of the deck (from June 2012) clearly states that these are preliminary results. The title paraphrases the conclusion to every other research study ever: “We’re on the right track, but much remains to be done.” This was an experiment in courier detection and a work in progress, and yet the two publications not only pretend that it was a deployed system, but also imply that the algorithm was used to generate a kill list for drone strokes. You can’t prove a negative of course, but there’s zero evidence here to substantiate the story.  In reality of course you would combine the results from this kind of analysis with other intelligence, which is exactly what the NSA do – another slide shows that ‘courier machine learning models’ are just one small component of a much larger suite of data analytics used to identify targets, as you’d expect. And of course data analytics will in turn be just one part of a broader intelligence processing effort. Nobody is being killed because of a flaky algorithm. The NSA couldn’t be that stupid and still actually be capable of finding Pakistan on a map. It’s a shame, because there’s a lot to pick apart in this story, from ethical questions about bulk data gathering and tracking journalists to technical ones. Realistically, how well can you evaluate an algorithm when the original data contains so many people whose classification is unknown? And is ‘courier’ a clear cut category to begin with, or an ever-changing ‘fuzzy’ set? Finally, it’s a great example of why often the most important thing in artificial intelligence isn’t the fancy algorithms you use but having a really well-defined and well-understood question to start with. It’s only when you fully understand the question that you can truly evaluate the results, as Ars Technica and The Intercept have neatly demonstrated. q0X~  The 18th of March 2018, was the day tech insiders had been dreading. That night, a new moon added almost no light to a poorly lit four-lane road in Tempe, Arizona, as a specially adapted Uber Volvo XC90 detected an object ahead. Part of the modern gold rush to develop self-driving vehicles, the SUV had been driving autonomously, with no input from its human backup driver, for 19 minutes. An array of radar and light-emitting lidar sensors allowed onboard algorithms to calculate that, given their host vehicle’s steady speed of 43mph, the object was six seconds away – assuming it remained stationary. But objects in roads seldom remain stationary, so more algorithms crawled a database of recognizable mechanical and biological entities, searching for a fit from which this one’s likely behavior could be inferred. At first the computer drew a blank; seconds later, it decided it was dealing with another car, expecting it to drive away and require no special action. Only at the last second was a clear identification found – a woman with a bike, shopping bags hanging confusingly from handlebars, doubtless assuming the Volvo would route around her as any ordinary vehicle would. Barred from taking evasive action on its own, the computer abruptly handed control back to its human master, but the master wasn’t paying attention. Elaine Herzberg, aged 49, was struck and killed, leaving more reflective members of the tech community with two uncomfortable questions: was this algorithmic tragedy inevitable? And how used to such incidents would we, should we, be prepared to get? “In some ways we’ve lost agency. When programs pass into code and code passes into algorithms and then algorithms start to create new algorithms, it gets farther and farther from human agency. Software is released into a code universe which no one can fully understand.” If these words sound shocking, they should, not least because Ellen Ullman, in addition to having been a distinguished professional programmer since the 1970s, is one of the few people to write revealingly about the process of coding. There’s not much she doesn’t know about software in the wild. “People say, ‘Well, what about Facebook – they create and use algorithms and they can change them.’ But that’s not how it works. They set the algorithms off and they learn and change and run themselves. Facebook intervene in their running periodically, but they really don’t control them. And particular programs don’t just run on their own, they call on libraries, deep operating systems and so on ...” Few subjects are more constantly or fervidly discussed right now than algorithms. But what is an algorithm? In fact, the usage has changed in interesting ways since the rise of the internet – and search engines in particular – in the mid-1990s. At root, an algorithm is a small, simple thing; a rule used to automate the treatment of a piece of data. If a happens, then do b; if not, then do c. This is the “if/then/else” logic of classical computing. If a user claims to be 18, allow them into the website; if not, print “Sorry, you must be 18 to enter”. At core, computer programs are bundles of such algorithms. Recipes for treating data. On the micro level, nothing could be simpler. If computers appear to be performing magic, it’s because they are fast, not intelligent. Recent years have seen a more portentous and ambiguous meaning emerge, with the word “algorithm” taken to mean any large, complex decision-making software system; any means of taking an array of input – of data – and assessing it quickly, according to a given set of criteria (or “rules”). This has revolutionized areas of medicine, science, transport, communication, making it easy to understand the utopian view of computing that held sway for many years. Algorithms have made our lives better in myriad ways. Only since 2016 has a more nuanced consideration of our new algorithmic reality begun to take shape. If we tend to discuss algorithms in almost biblical terms, as independent entities with lives of their own, it’s because we have been encouraged to think of them in this way. Corporations like Facebook and Google have sold and defended their algorithms on the promise of objectivity, an ability to weigh a set of conditions with mathematical detachment and absence of fuzzy emotion. No wonder such algorithmic decision-making has spread to the granting of loans/ bail/benefits/college places/job interviews and almost anything requiring choice. We no longer accept the sales pitch for this type of algorithm so meekly. In her 2016 book Weapons of Math Destruction, Cathy O’Neil, a former math prodigy who left Wall Street to teach and write and run the excellent mathbabe blog, demonstrated beyond question that, far from eradicating human biases, algorithms could magnify and entrench them. After all, software is written by overwhelmingly affluent white and Asian men – and it will inevitably reflect their assumptions (Google “racist soap dispenser” to see how this plays out in even mundane real-world situations). Bias doesn’t require malice to become harm, and unlike a human being, we can’t easily ask an algorithmic gatekeeper to explain its decision. O’Neil called for “algorithmic audits” of any systems directly affecting the public, a sensible idea that the tech industry will fight tooth and nail, because algorithms are what the companies sell; the last thing they will volunteer is transparency. The good news is that this battle is under way. The bad news is that it’s already looking quaint in relation to what comes next. So much attention has been focused on the distant promises and threats of artificial intelligence, AI, that almost no one has noticed us moving into a new phase of the algorithmic revolution that could be just as fraught and disorienting – with barely a question asked. The algorithms flagged by O’Neil and others are opaque but predictable: they do what they’ve been programmed to do. A skilled coder can in principle examine and challenge their underpinnings. Some of us dream of a citizen army to do this work, similar to the network of amateur astronomers who support professionals in that field. Legislation to enable this seems inevitable. We might call these algorithms “dumb”, in the sense that they’re doing their jobs according to parameters defined by humans. The quality of result depends on the thought and skill with which they were programmed. At the other end of the spectrum is the more or less distant dream of human-like artificial general intelligence, or AGI. A properly intelligent machine would be able to question the quality of its own calculations, based on something like our own intuition (which we might think of as a broad accumulation of experience and knowledge). To put this into perspective, Google’s DeepMind division has been justly lauded for creating a program capable of mastering arcade games, starting with nothing more than an instruction to aim for the highest possible score. This technique is called “reinforcement learning” and works because a computer can play millions of games quickly in order to learn what generates points. Some call this form of ability “artificial narrow intelligence”, but here the word “intelligent” is being used much as Facebook uses “friend” – to imply something safer and better understood than it is. Why? Because the machine has no context for what it’s doing and can’t do anything else. Neither, crucially, can it transfer knowledge from one game to the next (so-called “transfer learning”), which makes it less generally intelligent than a toddler, or even a cuttlefish. We might as well call an oil derrick or an aphid “intelligent”. Computers are already vastly superior to us at certain specialized tasks, but the day they rival our general ability is probably some way off – if it ever happens. Human beings may not be best at much, but we’re second-best at an impressive range of things. Here’s the problem. Between the “dumb” fixed algorithms and true AI lies the problematic halfway house we’ve already entered with scarcely a thought and almost no debate, much less agreement as to aims, ethics, safety, best practice. If the algorithms around us are not yet intelligent, meaning able to independently say “that calculation/course of action doesn’t look right: I’ll do it again”, they are nonetheless starting to learn from their environments. And once an algorithm is learning, we no longer know to any degree of certainty what its rules and parameters are. At which point we can’t be certain of how it will interact with other algorithms, the physical world, or us. Where the “dumb” fixed algorithms – complex, opaque and inured to real time monitoring as they can be – are in principle predictable and interrogable, these ones are not. After a time in the wild, we no longer know what they are: they have the potential to become erratic. We might be tempted to call these “frankenalgos” – though Mary Shelley couldn’t have made this up. These algorithms are not new in themselves. I first encountered them almost five years ago while researching a piece for the Guardian about high frequency trading (HFT) on the stock market. What I found was extraordinary: a human-made digital ecosystem, distributed among racks of black boxes crouched like ninjas in billion-dollar data farms – which is what stock markets had become. Where once there had been a physical trading floor, all action had devolved to a central server, in which nimble, predatory algorithms fed off lumbering institutional ones, tempting them to sell lower and buy higher by fooling them as to the state of the market. Human HFT traders (although no human actively traded any more) called these large, slow participants “whales”, and they mostly belonged to mutual and pension funds – ie the public. For most HFT shops, whales were now the main profit source. In essence, these algorithms were trying to outwit each other; they were doing invisible battle at the speed of light, placing and cancelling the same order 10,000 times per second or slamming so many into the system that the whole market shook – all beyond the oversight or control of humans. No one could be surprised that this situation was unstable. A “flash crash” had occurred in 2010, during which the market went into freefall for five traumatic minutes, then righted itself over another five – for no apparent reason. I travelled to Chicago to see a man named Eric Hunsader, whose prodigious programming skills allowed him to see market data in far more detail than regulators, and he showed me that by 2014, “mini flash crashes” were happening every week. Even he couldn’t prove exactly why, but he and his staff had begun to name some of the “algos” they saw, much as crop circle hunters named the formations found in English summer fields, dubbing them “Wild Thing”, “Zuma”, “The Click” or “Disruptor”. Neil Johnson, a physicist specializing in complexity at George Washington University, made a study of stock market volatility. “It’s fascinating,” he told me. “I mean, people have talked about the ecology of computer systems for years in a vague sense, in terms of worm viruses and so on. But here’s a real working system that we can study. The bigger issue is that we don’t know how it’s working or what it could give rise to. And the attitude seems to be ‘out of sight, out of mind’.” Facebook would claim they know what’s going on at the micro level … But what happens at the level of the population​? Significantly, Johnson’s paper on the subject was published in the journal Nature and described the stock market in terms of “an abrupt system-wide transition from a mixed human-machine phase to a new all-machine phase characterized by frequent black swan [ie highly unusual] events with ultrafast durations”. The scenario was complicated, according to the science historian George Dyson, by the fact that some HFT firms were allowing the algos to learn – “just letting the black box try different things, with small amounts of money, and if it works, reinforce those rules. We know that’s been done. Then you actually have rules where nobody knows what the rules are: the algorithms create their own rules – you let them evolve the same way nature evolves organisms.” Non-finance industry observers began to postulate a catastrophic global “splash crash”, while the fastest-growing area of the market became (and remains) instruments that profit from volatility. In his 2011 novel The Fear Index, Robert Harris imagines the emergence of AGI – of the Singularity, no less – from precisely this digital ooze. To my surprise, no scientist I spoke to would categorically rule out such a possibility. All of which could be dismissed as high finance arcana, were it not for a simple fact. Wisdom used to hold that technology was adopted first by the porn industry, then by everyone else. But the 21st century’s porn is finance, so when I thought I saw signs of HFT-like algorithms causing problems elsewhere, I called Neil Johnson again.  “You’re right on point,” he told me: a new form of algorithm is moving into the world, which has “the capability to rewrite bits of its own code”, at which point it becomes like “a genetic algorithm”. He thinks he saw evidence of them on fact-finding forays into Facebook (“I’ve had my accounts attacked four times,” he adds). If so, algorithms are jousting there, and adapting, as on the stock market. “After all, Facebook is just one big algorithm,” Johnson says.  “And I think that’s exactly the issue Facebook has. They can have simple algorithms to recognize my face in a photo on someone else’s page, take the data from my profile and link us together. That’s a very simple concrete algorithm. But the question is what is the effect of billions of such algorithms working together at the macro level? You can’t predict the learned behavior at the level of the population from microscopic rules. So Facebook would claim that they know exactly what’s going on at the micro level, and they’d probably be right. But what happens at the level of the population? That’s the issue.” To underscore this point, Johnson and a team of colleagues from the University of Miami and Notre Dame produced a paper, Emergence of Extreme Subpopulations from Common Information and Likely Enhancement from Future Bonding Algorithms, purporting to mathematically prove that attempts to connect people on social media inevitably polarize society as a whole. He thinks Facebook and others should model (or be made to model) the effects of their algorithms in the way climate scientists model climate change or weather patterns. O’Neil says she consciously excluded this adaptive form of algorithm from Weapons of Math Destruction. In a convoluted algorithmic environment where nothing is clear, apportioning responsibility to particular segments of code becomes extremely difficult. This makes them easier to ignore or dismiss, because they and their precise effects are harder to identify, she explains, before advising that if I want to see them in the wild, I should ask what a flash crash on Amazon might look like. “I’ve been looking out for these algorithms, too,” she says, “and I’d been thinking: ‘Oh, big data hasn’t gotten there yet.’ But more recently a friend who’s a bookseller on Amazon has been telling me how crazy the pricing situation there has become for people like him. Every so often you will see somebody tweet ‘Hey, you can buy a luxury yarn on Amazon for $40,000.’ And whenever I hear that kind of thing, I think: ‘Ah! That must be the equivalent of a flash crash!’” Anecdotal evidence of anomalous events on Amazon is plentiful, in the form of threads from bemused sellers, and at least one academic paper from 2016, which claims: “Examples have emerged of cases where competing pieces of algorithmic pricing software interacted in unexpected ways and produced unpredictable prices, as well as cases where algorithms were intentionally designed to implement price fixing.” The problem, again, is how to apportion responsibility in a chaotic algorithmic environment where simple cause and effect either doesn’t apply or is nearly impossible to trace. As in finance, deniability is baked into the system. Where safety is at stake, this really matters. When a driver ran off the road and was killed in a Toyota Camry after appearing to accelerate wildly for no obvious reason, Nasa experts spent six months examining the millions of lines of code in its operating system, without finding evidence for what the driver’s family believed had occurred, but the manufacturer steadfastly denied – that the car had accelerated of its own accord. Only when a pair of embedded software experts spent 20 months digging into the code were they able to prove the family’s case, revealing a twisted mass of what programmers call “spaghetti code”, full of algorithms that jostled and fought, generating anomalous, unpredictable output. The autonomous cars currently being tested may contain 100m lines of code and, given that no programmer can anticipate all possible circumstances on a real-world road, they have to learn and receive constant updates. How do we avoid clashes in such a fluid code milieu, not least when the algorithms may also have to defend themselves from hackers? You have all these pieces of code running on people’s iPhones, and collectively it acts like one multicellular organism Twenty years ago, George Dyson anticipated much of what is happening today in his classic book Darwin Among the Machines. The problem, he tells me, is that we’re building systems that are beyond our intellectual means to control. We believe that if a system is deterministic (acting according to fixed rules, this being the definition of an algorithm) it is predictable – and that what is predictable can be controlled. Both assumptions turn out to be wrong. “It’s proceeding on its own, in little bits and pieces,” he says. “What I was obsessed with 20 years ago that has completely taken over the world today are multicellular, metazoan digital organisms, the same way we see in biology, where you have all these pieces of code running on people’s iPhones, and collectively it acts like one multicellular organism. “There’s this old law called Ashby’s law that says a control system has to be as complex as the system it’s controlling, and we’re running into that at full speed now, with this huge push to build self-driving cars where the software has to have a complete model of everything, and almost by definition we’re not going to understand it. Because any model that we understand is gonna do the thing like run into a fire truck ’cause we forgot to put in the fire truck.” Unlike our old electro-mechanical systems, these new algorithms are also impossible to test exhaustively. Unless and until we have super-intelligent machines to do this for us, we’re going to be walking a tightrope. Dyson questions whether we will ever have self-driving cars roaming freely through city streets, while Toby Walsh, a professor of artificial intelligence at the University of New South Wales who wrote his first program at age 13 and ran a tyro computing business by his late teens, explains from a technical perspective why this is. “No one knows how to write a piece of code to recognize a stop sign. We spent years trying to do that kind of thing in AI – and failed! It was rather stalled by our stupidity, because we weren’t smart enough to learn how to break the problem down. You discover when you program that you have to learn how to break the problem down into simple enough parts that each can correspond to a computer instruction [to the machine]. We just don’t know how to do that for a very complex problem like identifying a stop sign or translating a sentence from English to Russian – it’s beyond our capability. All we know is how to write a more general purpose algorithm that can learn how to do that given enough examples.” Hence the current emphasis on machine learning. We now know that Herzberg, the pedestrian killed by an automated Uber car in Arizona, died because the algorithms wavered in correctly categorizing her. Was this a result of poor programming, insufficient algorithmic training or a hubristic refusal to appreciate the limits of our technology? The real problem is that we may never know. “And we will eventually give up writing algorithms altogether,” Walsh continues, “because the machines will be able to do it far better than we ever could. Software engineering is in that sense perhaps a dying profession. It’s going to be taken over by machines that will be far better at doing it than we are.” Walsh believes this makes it more, not less, important that the public learn about programming, because the more alienated we become from it, the more it seems like magic beyond our ability to affect. When shown the definition of “algorithm” given earlier in this piece, he found it incomplete, commenting: “I would suggest the problem is that algorithm now means any large, complex decision making software system and the larger environment in which it is embedded, which makes them even more unpredictable.” A chilling thought indeed. Accordingly, he believes ethics to be the new frontier in tech, foreseeing “a golden age for philosophy” – a view with which Eugene Spafford of Purdue University, a cybersecurity expert, concurs. “Where there are choices to be made, that’s where ethics comes in. And we tend to want to have an agency that we can interrogate or blame, which is very difficult to do with an algorithm. This is one of the criticisms of these systems so far, in that it’s not possible to go back and analyze exactly why some decisions are made, because the internal number of choices is so large that how we got to that point may not be something we can ever recreateto prove culpability beyond doubt.” The counter-argument is that, once a program has slipped up, the entire population of programs can be rewritten or updated so it doesn’t happen again – unlike humans, whose propensity to repeat mistakes will doubtless fascinate intelligent machines of the future. Nonetheless, while automation should be safer in the long run, our existing system of tort law, which requires proof of intention or negligence, will need to be rethought. A dog is not held legally responsible for biting you; its owner might be, but only if the dog’s action is thought foreseeable. In an algorithmic environment, many unexpected outcomes may not have been foreseeable to humans – a feature with the potential to become a scoundrel’s charter, in which deliberate obfuscation becomes at once easier and more rewarding. Pharmaceutical companies have benefited from the cover of complexity for years (see the case of Thalidomide), but here the consequences could be both greater and harder to reverse. Commerce, social media, finance and transport may come to look like small beer in future, however. If the military no longer drives innovation as it once did, it remains tech’s most consequential adopter. No surprise, then, that an outpouring of concern among scientists and tech workers has accompanied revelations that autonomous weapons are ghosting toward the battlefield in what amounts to an algorithmic arms race. A robotic sharpshooter currently polices the demilitarized zone between North and South Korea, and while its manufacturer, Samsung, denies it to be capable of autonomy, this claim is widely disbelieved. Russia, China and the US all claim to be at various stages of developing swarms of coordinated, weaponized drones , while the latter plans missiles able to hover over a battlefield for days, observing, before selecting their own targets. A group of Google employees resigned over and thousands more questioned the tech monolith’s provision of machine learning software to the Pentagon’s Project Maven “algorithmic warfare” program – concerns to which management eventually responded, agreeing not to renew the Maven contract and to publish a code of ethics for the use of its algorithms. At time of writing, competitors including Amazon and Microsoft have resisted following suit. In common with other tech firms, Google had claimed moral virtue for its Maven software: that it would help choose targets more efficiently and thereby save lives. The question is how tech managers can presume to know what their algorithms will do or be directed to do in situ – especially given the certainty that all sides will develop adaptive algorithmic counter-systems designed to confuse enemy weapons. As in the stock market, unpredictability is likely to be seen as an asset rather than handicap, giving weapons a better chance of resisting attempts to subvert them. In this and other ways we risk in effect turning our machines inside out, wrapping our everyday corporeal world in spaghetti code. Lucy Suchman of Lancaster University in the UK co-authored an open letter from technology researchers to Google, asking them to reflect on the rush to militarize their work. Tech firms’ motivations are easy to fathom, she says: military contracts have always been lucrative. For the Pentagon’s part, a vast network of sensors and surveillance systems has run ahead of any ability to use the screeds of data so acquired. “They are overwhelmed by data, because they have new means to collect and store it, but they can’t process it. So it’s basically useless – unless something magical happens. And I think their recruitment of big data companies is a form of magical thinking in the sense of: ‘Here is some magic technology that will make sense of all this.’” Suchman also offers statistics that shed chilling light on Maven. According to analysis carried out on drone attacks in Pakistan from 2003-13, fewer than 2% of people killed in this way are confirmable as “high value” targets presenting a clear threat to the United States. In the region of 20% are held to be non-combatants, leaving more than 75% unknown. Even if these figures were out by a factor of two – or three, or four – they would give any reasonable person pause. “So here we have this very crude technology of identification and what Project Maven proposes to do is automate that. At which point it becomes even less accountable and open to questioning. It’s a really bad idea.” Suchman’s colleague Lilly Irani, at the University of California, San Diego, reminds us that information travels around an algorithmic system at the speed of light, free of human oversight. Technical discussions are often used as a smokescreen to avoid responsibility, she suggests. “When we talk about algorithms, sometimes what we’re talking about is bureaucracy. The choices algorithm designers and policy experts make are presented as objective, where in the past someone would have had to take responsibility for them. Tech companies say they’re only improving accuracy with Maven – ie the right people will be killed rather than the wrong ones – and in saying that, the political assumption that those people on the other side of the world are more killable, and that the US military gets to define what suspicion looks like, go unchallenged. So technology questions are being used to close off some things that are actually political questions. The choice to use algorithms to automate certain kinds of decisions is political too.” The legal conventions of modern warfare, imperfect as they might be, assume human accountability for decisions taken. At the very least, algorithmic warfare muddies the water in ways we may grow to regret. A group of government experts is debating the issue at the UN convention on certain conventional weapons (CCW) meeting in Geneva this week. Solutions exist or can be found for most of the problems described here, but not without incentivizing big tech to place the health of society on a par with their bottom lines. More serious in the long term is growing conjecture that current programming methods are no longer fit for purpose given the size, complexity and interdependency of the algorithmic systems we increasingly rely on. One solution, employed by the Federal Aviation Authority in relation to commercial aviation, is to log and assess the content of all programs and subsequent updates to such a level of detail that algorithmic interactions are well understood in advance – but this is impractical on a large scale. Portions of the aerospace industry employ a relatively new approach called model-based programming, in which machines do most of the coding work and are able to test as they go. Model-based programming may not be the panacea some hope for, however. Not only does it push humans yet further from the process, but Johnson, the physicist, conducted a study for the Department of Defense that found “extreme behaviors that couldn’t be deduced from the code itself” even in large, complex systems built using this technique. Much energy is being directed at finding ways to trace unexpected algorithmic behavior back to the specific lines of code that caused it. No one knows if a solution (or solutions) will be found, but none are likely to work where aggressive algos are designed to clash and/or adapt. As we wait for a technological answer to the problem of soaring algorithmic entanglement, there are precautions we can take. Paul Wilmott, a British expert in quantitative analysis and vocal critic of high frequency trading on the stock market, wryly suggests “learning to shoot, make jam and knit”. More practically, Spafford, the software security expert, advises making tech companies responsible for the actions of their products, whether specific lines of rogue code – or proof of negligence in relation to them – can be identified or not. He notes that the venerable Association for Computing Machinery has updated its code of ethics along the lines of medicine’s Hippocratic oath, to instruct computing professionals to do no harm and consider the wider impacts of their work. Johnson, for his part, considers our algorithmic discomfort to be at least partly conceptual; growing pains in a new realm of human experience. He laughs in noting that when he and I last spoke about this stuff a few short years ago, my questions were niche concerns, restricted to a few people who pored over the stock market in unseemly detail. “And now, here we are – it’s even affecting elections. I mean, what the heck is going on? I think the deep scientific thing is that software engineers are trained to write programs to do things that optimize – and with good reason, because you’re often optimizing in relation to things like the weight distribution in a plane, or a most fuel-efficient speed: in the usual, anticipated circumstances optimizing makes sense. But in unusual circumstances it doesn’t, and we need to ask: ‘What’s the worst thing that could happen in this algorithm once it starts interacting with others?’ The problem is we don’t even have a word for this concept, much less a science to study it.” He pauses for moment, trying to wrap his brain around the problem. “The thing is, optimizing is all about either maximizing or minimizing something, which in computer terms are the same. So what is the opposite of an optimization, ie the least optimal case, and how do we identify and measure it? The question we need to ask, which we never do, is: ‘What’s the most extreme possible behavior in a system I thought I was optimizing?’” Another brief silence ends with a hint of surprise in his voice. “Basically, we need a new science,” he says. Andrew Smith’s Totally Wired: The Rise and Fall of Joshua Harris and the Great Dotcom Swindle will be published by Grove Atlantic next Februaryq1X�   It was the case of the missing PhD student.  As another academic year got under way at Imperial College London, a senior professor was bemused at the absence of one of her students. He had worked in her lab for three years and had one more left to complete his studies. But he had stopped coming in. Eventually, the professor called him. He had left for a six-figure salary at Apple. “He was offered such a huge amount of money that he simply stopped everything and left,” said Maja Pantic, professor of affective and behavioural computing at Imperial. “It’s five times the salary I can offer. It’s unbelievable. We cannot compete.” It is not an isolated event. Across the country, talented computer scientists are being lured from academia by private sector offers that are hard to turn down. According to a Guardian survey of Britain’s top ranking research universities, tech firms are hiring AI experts at a prodigious rate, fuelling a brain drain that has already hit research and teaching. One university executive warned of a “missing generation” of academics who would normally teach students and be the creative force behind research projects.  The impact of the brain drain may reach far beyond academia. Pantic said the majority of top AI researchers moved to a handful of companies, meaning their skills and experience were not shared through society. “That’s a problem because only a diffusion of innovation, rather than its concentration into just a few companies, can mitigate the dramatic disruptions and negative effects that AI may bring about.” Artificial Intelligence has various definitions, but in general it means a program that uses data to build a model of some aspect of the world. This model is then used to make informed decisions and predictions about future events. The technology is used widely, to provide speech and face recognition, language translation, and personal recommendations on music, film and shopping sites. In the future, it could deliver driverless cars, smart personal assistants, and intelligent energy grids. AI has the potential to make organisations more effective and efficient, but the technology raises serious issues of ethics, governance, privacy and law.  She is concerned that major tech firms are creating a huge pay gap between AI professionals and the rest of the workforce. Beyond getting the companies to pay their taxes, Pantic said the government might have to consider pay caps, a strategy that has reined in corporate salaries in Nordic countries.  Many of the best researchers move to Google, Amazon, Facebook and Apple. “The creme de la creme of academia has been bought and that is worrying,” Pantic said. “If the companies don’t pay tax it’s a problem for the government. The government doesn’t get enough money to educate people, or to invest in academia. It’s a vicious circle.” When Murray Shanahan, another Imperial researcher, received a job offer from DeepMind, Google’s London-based artificial intelligence group, he thought hard about the decision. He saw plenty of positives to joining the company. It was a chance to pursue his work without the burden of other academic duties. He would have access to fabulous computing resources. And he would work alongside some of the best in the field. But despite the long list of pros, Shanahan paused. “The potential impact on academia of the current tech hiring frenzy was one of the issues that bothered me,” he said. Shanahan decided to negotiate a joint position. It allowed him to have a foot in both camps, keeping his chair at Imperial while becoming a senior scientist at DeepMind.  For those with the right skills, the hiring boom has obvious positives. Heavy investment from tech firms means there are many more jobs in artificial intelligence than there are qualified candidates. To recruit the best talent, companies offer high salaries, impressive computing facilities and technical challenges that have the potential to affect the lives of billions.  In the past, brilliant mathematicians, physicists and computer scientists headed to the City for serious money. Now they are as likely to train in AI and move to tech firms. “There are fantastic opportunities in industry, the sorts of opportunities that make working in the City seem really dull and not particularly well paid,” said Zoubin Ghahramani, professor of information engineering at Cambridge University and chief scientist at Uber, the ride-hailing firm. “It’s both intellectually interesting and, from a lifestyle point of view, very difficult to turn down.” Ghahramani announced his move to Uber in March. For now, he commutes to the company’s San Francisco office one week every month. Next summer, he will move to the city full time. Beyond the difference in salaries, he lists a host of other reasons that academics are lured into industry. University roles come with administrative duties that some find onerous: teaching, marking, being on committees, and the endless chasing of grants. In industry, star recruits can focus purely on their research. But there is more to it than that. The explosion of interest in artificial intelligence is driven by the success of machine learning, a field that uses algorithms to find meaningful patterns in data. To work well, many of today’s algorithms must be trained on huge amounts of data, a task that takes a lot of computer power. Without collaborative projects, universities can rarely compete with the big tech firms on data or computing. Instead, they focus on new ideas: building algorithms that learn from less information, for instance. Ghahramani began working at Uber part time when the company bought Geometric Intelligence, his AI startup, last year. As chief scientist, he will oversee the use of machine learning algorithms to understand how cities work and how people move around them. The end goal is to match the supply of rides to demand. “The interesting thing about this is we’re doing machine learning in the real, physical world of cities. We’re trying to optimise the movement of people and things around the world,” he said. Ghahramani sees no sign that industry’s demand for talented AI researchers has peaked. “It’s very fierce right now and it’s yet to show signs of tapering off,” he said. “Universities will have to train enough people to meet the demand, and that’s a challenge if lecturers and postdocs are being lured into industry. It’s like killing the geese that lay the golden eggs. Companies are starting to realise that and some of the major tech companies are starting to give back to universities by sponsoring lectureships and donating funds.” Steven Turner joined Amazon Web Services in Cambridge last year. He helps companies to build their own Amazon-style “recommendation engines”, and use image recognition, computer speech and chatbots in customer service. One financial institution he has worked with now uses technology to answer simple questions, such as on customer mortgage rates, freeing up humans for more complex queries.  In academia, he saw departments fighting for funds to continue their work and keep people from leaving. The main reason he left was to work on real problems rather than more theoretical concepts. But the culture at Amazon turned out to be more vibrant than in academia. At university, Turner found being a PhD student isolating at times, even though his supervisor was a brilliant mentor. “I personally think that having a greater focus on culture and social interaction to ensure researchers don’t feel as isolated as they can do would have a significant impact on retention,” Turner told the Guardian. He said universities should also focus on researchers’ career development, giving free access to external training and teaming up with business schools to broaden researchers’ knowledge. Ghahramani believes UK universities will have to become more flexible about researchers holding joint positions. “They need to be flexible about intellectual property arrangements. They need to be flexible about PhD students who might want to spend time in a world-leading industry AI lab. That’s what we need to get around the problems. The universities that have been flexible have benefited,” he said. q2e.