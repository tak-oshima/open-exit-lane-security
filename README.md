# Open Exit Lane Security Breach Control

Open Exit Lane Security Breach Control is a Python program for detecting objects traveling in the wrong direction at an airport open exit lane.

## Installation

Make sure you are running Python 3.7 or above.

Clone the project repository to your local machine.

```bash
$ git clone https://github.com/tkykntm/open-exit-lane-security
```

Move into the project directory and create a virtualenv.

```bash
$ cd open-exit-lane-security

# Mac OS
$ python3 -m venv env

# Windows
$ python -m venv env
```

Activate the virtualenv.

```bash
# Mac OS
$ source env/bin/activate

# Windows
$ .\env\Scripts\activate
```

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install dependencies listed in reqirements.txt.

```bash
$ pip install -r requirements.txt
```

## Usage

```bash
# Example usage
$ python bg_sub.py --input assets/people.mp4 --detector MOG2

'''
optional arguments:
  -h, --help           show this help message and exit.
  --input INPUT        path to a video or a sequence of image. Default is video device.
  --detector DETECTOR  background subtraction method (MOG, MOG2, GMG, KNN).
  --size SIZE          minimum pixel area of object to be detected.
  --breach-dir DIR     direction of security breach detection.
'''
```