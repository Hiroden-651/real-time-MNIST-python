# real-time-MNIST-python

This project uses live input from a camera in order to classify hand-written digits using a neural-network trained on the MNSIT data set. Input from the camera is transformed into a format understood by the network, which then classifies the digit as being in the category range of 0-9.

## Prerequisites

For software, this program is written using python 3 and uses the Keras, h5py, OpenCV(4.1.0), and Numpy libraries. I used [miniconda3](https://docs.conda.io/en/latest/miniconda.html) for installing the respective libraries although they should be able to be installed using [pip](https://pypi.org/project/pip/) as well.

For hardware, specifically the camera, I used a Logitech C930e(1080p@30fps maximum). This program should also work with built-in laptop webcams as long as the proper camera device is specified in the program. Video capture speed will largely be dependent on both physical hardware and the camera used. My setup consisted of the Logitech camera and a Thinkpad T430(i7-3632QM & 8GB DDR3 RAM) running Ubuntu 18.04 LTS.

## Usage

Once the libraries are installed, clone the repo and run with the following command:

'''bash
python main.py
'''

This creates two OpenCV image windows: A larger window showing the whole capture input with a window-of-interest specified by a color frame and text denoting classification, and a smaller window showing the MNIST form of the window-of-interest; this is classified by the neural network with the predicted digit written to the larger window.

To classify a digit, simply place a hand-written number(0-9) within the window-of-interest. Classifications are made every 15th frame with results being printed to the larger image window.

To close the program, press the 'Q' key.

## Contributing

This program was meant more of a personal exercise than an attempt at a professional product. That said, anyone can take this and use it as they see fit. Go wild.

This is also my first real non-academic project. I am welcoming of any suggestions/criticisms that can be used to improve the program and documentation(including this README).

## License

[MIT](https://opensource.org/licenses/MIT)


