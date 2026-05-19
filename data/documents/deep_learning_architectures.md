# Deep Learning Architectures and their Evolution

Deep learning is a subset of machine learning that is based on artificial neural networks with multiple layers. These "deep" layers enable the hierarchical extraction of features from raw data. Over the years, several architectural breakthroughs have propelled the field forward, leading to human-level performance in tasks ranging from image recognition to natural language processing.

## Feedforward Neural Networks (FNNs)

The simplest form of a neural network is the Feedforward Neural Network. In this architecture, information moves in only one direction—forward—from the input nodes, through the hidden nodes (if any), and to the output nodes. There are no cycles or loops in the network. FNNs are typically used for basic classification and regression tasks but struggle with complex data types like sequential data or high-resolution images due to their lack of spatial or temporal awareness.

## Convolutional Neural Networks (CNNs)

Convolutional Neural Networks revolutionised the field of computer vision. Introduced prominently in the 1990s by Yann LeCun and popularized by AlexNet in 2012, CNNs are designed specifically to process data that has a known grid-like topology, such as images.

**Key Components of CNNs:**
1. **Convolutional Layers:** Apply filters (kernels) to the input to detect features such as edges, textures, and patterns. These filters slide over the input image, preserving spatial relationships.
2. **Pooling Layers:** Reduce the spatial dimensions of the data (downsampling), which decreases the computational load and helps make the model robust to slight translations in the input. Max pooling is the most common technique.
3. **Fully Connected Layers:** At the end of the network, high-level reasoning is done via fully connected layers, similar to FNNs.

CNNs have applications beyond image classification, including object detection (YOLO, Faster R-CNN), image segmentation, and even some areas of natural language processing where local patterns are important.

## Recurrent Neural Networks (RNNs)

While CNNs are excellent for spatial data, Recurrent Neural Networks (RNNs) were designed for sequential data. An RNN processes sequences by iterating through the elements and maintaining a 'hidden state' that contains information relative to what it has seen so far. 

**The Vanishing Gradient Problem:**
Traditional RNNs suffer heavily from the vanishing gradient problem. When training on long sequences, the gradients used to update the network weights can become infinitesimally small, meaning the network forgets early parts of the sequence.

### Long Short-Term Memory Networks (LSTMs)

To solve the vanishing gradient problem, Hochreiter and Schmidhuber introduced LSTMs in 1997. LSTMs contain a more complex computational block called a "memory cell". 

Each memory cell has three gates:
- **Forget Gate:** Decides what information to discard from the cell state.
- **Input Gate:** Decides which values from the input to update the memory state.
- **Output Gate:** Decides what to output based on input and the memory of the cell.

LSTMs became the standard for tasks like machine translation, speech recognition, and time-series forecasting for many years.

## The Transformer Architecture

In 2017, researchers at Google published the seminal paper "Attention Is All You Need," introducing the Transformer architecture. This completely changed the trajectory of Natural Language Processing and AI in general.

Unlike RNNs, which process data sequentially, Transformers process all tokens in a sequence simultaneously. This allows for massive parallelisation during training, which is why modern Large Language Models (LLMs) can be trained on vast amounts of internet data.

**Core Mechanisms of Transformers:**
1. **Self-Attention Mechanism:** Allows the model to weigh the importance of different words in a sentence relative to each other, regardless of their positional distance. For example, in the sentence "The bank of the river," the attention mechanism helps the model understand that "bank" refers to land, not a financial institution, by attending to "river."
2. **Multi-Head Attention:** Instead of computing attention once, the Transformer computes it multiple times in parallel, allowing the model to focus on different aspects of the text simultaneously (e.g., grammatical structure vs. semantic meaning).
3. **Positional Encoding:** Since Transformers don't process data sequentially, they need a way to understand the order of words. Positional encodings are added to the input embeddings to inject information about the relative or absolute position of the tokens.

The Transformer architecture is the foundation of almost all modern Large Language Models, including BERT (encoder-only), GPT (decoder-only), and T5 (encoder-decoder). Its success has also spilled over into computer vision with the advent of Vision Transformers (ViTs).
