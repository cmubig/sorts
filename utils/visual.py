import glob 
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import torch 

from typing import List
from natsort import natsorted

class Visual:
    def __init__(self, outdir, full_screen: bool = False, background: str = 'white') -> None:
        """ Initializes the visualizations. """
        
        self.full_screen = full_screen
        self.background = background
        self.hh = []
        self.fig_count = 0
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)
        self.reset()
        
    def reset(self) -> None:
        self.fig = plt.figure(figsize=(10, 10))
        self.sp = self.fig.add_subplot(111)

        # plt.ion()
        # self.fig.show()
        self.sp.set_facecolor(f'xkcd:{self.background}')
        
        self.radar()
        self.runway()
    
        plt.xlabel("X (in Km)")
        plt.ylabel("Y (in Km)")

        if self.full_screen:
            self.fig.canvas.manager.full_screen_toggle()
    
    def radar(self) -> None:
        self.sp.plot(
            3.7*np.sin(np.linspace(0, 2*np.pi, 100)), 3.7*np.cos(np.linspace(0, 2*np.pi, 100)), 
            color='grey', linestyle='--')
        self.sp.text(2.7,2.7,'2 NM', color = 'grey')
        self.sp.text(-2.7,-2.7,'2 NM', color = 'grey')

        self.sp.plot(
            7.4*np.sin(np.linspace(0, 2*np.pi, 100)), 7.4*np.cos(np.linspace(0, 2*np.pi, 100)), 
            color='grey', linestyle='--')
        self.sp.text(5.2,5.2,'4 NM', color = 'grey')
        self.sp.text(-5.2,-5.2,'4 NM', color = 'grey')

        self.sp.plot(
            11.26*np.sin(np.linspace(0, 2*np.pi, 100)), 11.26*np.cos(np.linspace(0, 2*np.pi, 100)), 
            color='grey', linestyle='--')
        self.sp.text(8,8,'6 NM', color = 'grey')
        self.sp.text(-8,-8,'6 NM', color = 'grey')

    def runway(self) -> None:
        self.sp.plot(1.45, 0, color='cyan', marker='p', markersize=12, zorder=12)
        self.sp.plot(
            np.linspace(0, 1.45, 100), np.linspace(0, 0, 100), color="black", lw=14, alpha=1, 
            zorder=10, label='Runway 2')
        self.sp.plot(
            np.linspace(0, 1.45, 100), np.linspace(0, 0, 100), '--', color="white", lw=1, alpha=1, 
            zorder=10)
    
    def plot(self, agents, show: bool = False, show_tree: bool = True, pause: float = 1.0) -> None:
        for agent in agents:
            state = agent.state
            color = agent.color
            traj = torch.cat(agent.trajectory).numpy()
            ref = agent.reference_trajectory
            id_ = agent.id

            # plot first, last (x, y) and trajectory
            self.sp.plot(ref[:, 0], ref[:, 1] , color=color, linewidth=10, alpha=0.2, zorder=0)
            self.sp.plot(traj[:, 0], traj[:, 1], color=color, linestyle='-', linewidth=4)
            self.sp.plot(state[0, 0], state[0, 1], color=color, marker='o', markersize=6)
            self.sp.plot(state[-1, 0], state[-1, 1], color=color, marker='o', markersize=10)
            
            if show_tree and len(agent.tree) > 0:
                for tree in agent.trees:
                    tree = torch.cat(agent.tree).numpy()
                    self.sp.plot(tree[:, 0], tree[:, 1], color='limegreen', linestyle='-', markersize=4)
                tree = torch.cat(agent.tree).numpy()
                self.sp.plot(tree[:, 0], tree[:, 1], color='magenta', linestyle='-', markersize=4)

            # self.sp.plot(state[1:, 0], state[1:, 1], color=color, linestyle='-', linewidth=6)

            speed = str(int(np.linalg.norm(state[-1,:2]-state[-2,:2]) * 1943)) + "Knots \n"
            alt = str(int(state[-1, 2] * 3280.84)) + "MSL"
            self.sp.text(
                state[-1,0]+0.5, state[-1,1]+0.5, f'A{id_}\n {speed} {alt}', color=color,fontsize=8)

        if show:
            plt.show()
            plt.waitforbuttonpress()
            # plt.pause(pause)
        else:
            plt.savefig(f"{self.outdir}/{self.fig_count}.png", bbox_inches='tight', dpi=200)
            self.fig_count += 1
        
        plt.close()

    def save(self, num_episode: int) -> None:
        self.fig_count = 0
        imgs = natsorted(glob.glob(f"{self.outdir}/*.png"))
        with imageio.get_writer(f"{self.outdir}/ep-{num_episode}.gif", mode='I', duration=0.5) as writer:
            for img in imgs:
                writer.append_data(imageio.imread(img))
                os.remove(img)

        