# 4900

use vmas to create a new scenario which is suitable for plane and tower environment

# Scenario

Agents are one plane and 2 towers, landmarks are start place, end place. a plane flies from start place to end place, towers speak and plane receives signal in tower's control area. Tower A's control area radius is 4, tower B's control area radius is 4,  tower A's control area and tower B's control area overlap. The origin (0, 0) changes according to which tower's control area the plane is in, at first, origin (0, 0) is start, then tower A's place, once going to  tower B's control area, origin (0, 0) is tower B's place. Plane can choose listen to closest tower. 



# Initialization

! git clone https://github.com/proroklab/VectorizedMultiAgentSimulator.git

%cd /content/VectorizedMultiAgentSimulator

!pip install -e .

!sudo apt-get update

!sudo apt-get install python3-opengl xvfb

!pip install pyvirtualdisplay

import pyvirtualdisplay

display = pyvirtualdisplay.Display(visible=False, size=(1400, 900))

display.start()

# Citation

initialization code comes from https://github.com/proroklab/VectorizedMultiAgentSimulator

I create my plane scenario, by extending the BaseScenario class in scenario.py which comes from https://github.com/proroklab/VectorizedMultiAgentSimulator

run.py uses part of code from https://github.com/proroklab/VectorizedMultiAgentSimulator
