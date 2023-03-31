import glob 
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import torch 

from typing import List
from natsort import natsorted
from PIL import Image

class Visual:
    def __init__(self, outdir, full_screen: bool = False, background: str = 'white') -> None:
        """ Initializes the visualization class. 
        
        Inputs
        ------
            outdir[str]: output directory path.
            full_screen[bool]: toggles full-screen if True
            background[str]: sets the color of the background. 
        """
        self.full_screen = full_screen
        self.background = background
        self.hh = []
        self.fig_count = 0
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)
        self.reset()
        
    def reset(self) -> None:
        """ Resets the visualization radar. """
        self.fig = plt.figure(figsize=(10, 10))

        self.sp = self.fig.add_subplot(111)
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
    
    def plot(self, agents, show: bool = False, show_tree: bool = True, agent_id = None) -> None:
        for agent in agents:
            state = agent.state
            color = agent.color

            first_state = agent.trajectory[0].numpy()
            last_state = agent.trajectory[-1].numpy()
            cur_traj = torch.cat(agent.trajectory).numpy()
            ref_traj = agent.reference_trajectory
            id_ = agent.id
            
            alpha, alpha_ref = 0.2, 0.1
            text = f'A{id_}\n Done!'
            if not agent.done:
                alpha, alpha_ref = 1.0, 0.2

                speed = str(int(np.linalg.norm(last_state[-1,:2]-last_state[-2,:2]) * 1943)) + "Knots \n"
                alt = str(int(last_state[-1, 2] * 3280.84)) + "MSL"
                text = f'A{id_}\n {speed} {alt}'
            self.sp.text(first_state[0,0]+0.5, first_state[0,1]+0.5, text, color=color, fontsize=8)

            # reference trajectory 
            self.sp.plot(ref_traj[:, 0], ref_traj[:, 1] , color=color, linewidth=10, alpha=alpha_ref, zorder=0)
            
            # executed trajectory so far:
            self.sp.plot(cur_traj[:, 0], cur_traj[:, 1], color=color, linestyle='-', linewidth=4, alpha=alpha)
            
            # markers for the start and end of last agent's state:
            self.sp.plot(last_state[0, 0], last_state[0, 1], color=color, marker='o', markersize=6, alpha=alpha)
            self.sp.plot(last_state[-1, 0], last_state[-1, 1], color=color, marker='o', markersize=10, alpha=alpha)

            # plot tree expansions
            if show_tree and len(agent.tree) > 0:
                if not agent_id is None and agent_id == id_:
                    for tree in agent.trees:
                        tree = torch.cat(agent.tree).numpy()
                        self.sp.plot(tree[:, 0], tree[:, 1], color='limegreen', linestyle='-', linewidth=3, markersize=4)
                
                    tree = torch.cat(agent.tree).numpy()
                    self.sp.plot(tree[:, 0], tree[:, 1], color='magenta', linestyle='-', linewidth=2, markersize=4)

                # self.sp.plot(state[:, 0], state[:, 1], color='magenta', linestyle='-', linewidth=6)

        if show:
            plt.show()
            plt.waitforbuttonpress()
            # plt.pause(pause)
        else:
            plt.savefig(f"{self.outdir}/{self.fig_count}.png", bbox_inches='tight', dpi=100)
            self.fig_count += 1
        
        plt.close()

    def save(self, num_episode: int) -> None:
        self.fig_count = 0
        imgs = natsorted(glob.glob(f"{self.outdir}/*.png"))
        with imageio.get_writer(f"{self.outdir}/ep-{num_episode}.gif", mode='I', duration=0.1) as writer:
            for img in imgs:
                writer.append_data(imageio.imread(img))
                os.remove(img)

        