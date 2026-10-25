# Neural Style Transfer using VGG Network

This project implements **Neural Style Transfer** using TensorFlow and Keras. The algorithm takes a content image ($C$) and a style image ($S$) to synthesize a new generated image ($G$) that blends the content of the former with the artistic style of the latter.

## Core Architecture & Workflow

1. **Pre-trained VGG Network**:
   - Uses a pre-trained VGG-19/VGG-16 architecture to extract high-level and low-level feature representations from images.

2. **Content Cost (`compute_content_cost`)**:
   - Measures how much the content of the generated image ($G$) differs from the content image ($C$) using hidden layer activations.
   - Unrolls feature maps and calculates the squared difference norm.

3. **Style Cost (`compute_style_cost` & `gram_matrix`)**:
   - Captures the style by computing the **Gram matrix** ($AA^T$) of feature representations across multiple chosen layers ($STYLE\_LAYERS$).
   - Quantifies the correlation between different feature channels to match textures and artistic patterns.

4. **Optimization & Training Loop (`total_cost` & `train_step`)**:
   - Combines content and style losses into a total cost function: $J(G) = \alpha J_{content}(C, G) + \beta J_{style}(S, G)$.
   - Uses the **Adam optimizer** and `tf.GradientTape` to iteratively update and optimize the pixel values of the generated image directly.

## Required Libraries

- TensorFlow (v2.x)
- NumPy
- Keras
- Matplotlib
- PIL (Pillow)
