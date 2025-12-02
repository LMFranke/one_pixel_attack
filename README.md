# One Pixel Attack Study

This project explores the sensitivity of the ResNet18 computer vision model by applying the Differential Evolution algorithm. It demonstrates how altering a limited number of pixels (specifically one pixel by default) can change the classification output of a deep learning model.

The tool uses **PyTorch** and **torchvision** to load a pre-trained ResNet18 and modifies the input image to minimize the confidence of the original target class.

## Features

- **Differential Evolution Algorithm:** Implements a population-based optimization method to find the most impactful pixel coordinates and RGB values.
- **ResNet18 Integration:** Uses the standard pre-trained ResNet18 model from `torchvision`.
- **GPU Acceleration:** Automatically detects and utilizes CUDA-compatible GPUs for faster processing.
- **High-Resolution Output:** Applies the calculated pixel modification back to the original high-resolution image and saves the result.

## Hardware Requirements

* **GPU Recommendation:** A high-performance NVIDIA GPU (**RTX 40 Series or 50 Series**) is strongly recommended.
* **Warning:** Running this algorithm on a CPU or older hardware can be extremely slow due to the high computational cost of evaluating thousands of potential pixel perturbations iteratively.

## Prerequisites

To run this project, you need Python installed along with the following libraries:

* Python 3.8+
* PyTorch (with CUDA support)
* Torchvision
* Pillow (PIL)

## Usage

1.  **Prepare your Image:**
    Place the image you want to test in a known directory.

2.  **Configure the Path:**
    Open `one_pixel_attack.py` and modify the `IMAGE_PATH` variable to point to your image file.
    
    ```python
    # In one_pixel_attack.py
    IMAGE_NAME = "your_image.jpg"
    IMAGE_PATH = "path/to/your/image/" + IMAGE_NAME
    ```

3.  **Run the Script:**
    Execute the Python script:
    ```bash
    python one_pixel_attack.py
    ```

4.  **Check Results:**
    * The console will display the optimization progress (generations).
    * Once a solution is found (or max generations reached), the modified image will be saved as `_attacked.png` in the same directory.
    * The script prints the new RGB values and coordinates of the modified pixel.

## Configuration

You can adjust the parameters inside the `run_attack_params` function call in the `__main__` block:

* `pixels`: Number of pixels to modify (Default: 1).
* `maxiter`: Maximum number of generations for the Differential Evolution algorithm (Default: 100).
* `popsize`: Population size for the algorithm (Default: 400).

## License

This project is open-source and available for educational and research purposes.
