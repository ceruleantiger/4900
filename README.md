# 4900

use vmas to create a new scenario which is suitable for plane and tower environment

Initialization

! git clone https://github.com/proroklab/VectorizedMultiAgentSimulator.git

%cd /content/VectorizedMultiAgentSimulator

!pip install -e .

!sudo apt-get update

!sudo apt-get install python3-opengl xvfb

!pip install pyvirtualdisplay

import pyvirtualdisplay

display = pyvirtualdisplay.Display(visible=False, size=(1400, 900))

display.start()

Citation

initialization, base scenario code comes from https://github.com/proroklab/VectorizedMultiAgentSimulator
