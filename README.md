# Reinforcement Learning SARSA
#To run Sarsa Agent:
python gridworldSarsa.py -a q -k 5
python pacman.py -p PacmanSarsaAgent -x 2000 -n 2010 -l smallGrid
python pacman.py -p ApproximateSarsaAgent -x 2000 -n 2010 -l smallGrid
python crawlerSarsa.py
#To run Q-learning Agent:
python gridworld.py -a q -k 5
python pacman.py -p PacmanQAgent -x 2000 -n 2010 -l smallGrid
python pacman.py -p ApproximateQAgent -x 2000 -n 2010 -l smallGrid
python crawler.py
