import torchvision.transforms as T
from PIL import Image
import gym
import numpy as np
import torch
import agents.dqn_agent as dqn_agent
from itertools import count
import matplotlib.pyplot as plt

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


class CartPoleAndDQN(object):
    def __init__(self):
        self.env = gym.make('CartPole-v0').unwrapped
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env.reset()

        init_screen = self.get_screen()
        _, _, screen_height, screen_width = init_screen.shape
        n_actions = self.env.action_space.n
        self.dqn_agent = dqn_agent.DQNAgent(screen_height, screen_width, n_actions, self.device)
        self.episode_durations = []

    def run(self):
        TARGET_UPDATE = 10
        num_episodes = 1000
        for i_episode in range(num_episodes):
            # Initialize the environment and state
            self.env.reset()
            last_screen = self.get_screen()
            current_screen = self.get_screen()
            state = current_screen - last_screen
            for t in count():
                # Select and perform an action
                action = self.dqn_agent.select_action(state)
                _, reward, done, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device)

                # Observe new state
                last_screen = current_screen
                current_screen = self.get_screen()
                if not done:
                    next_state = current_screen - last_screen
                else:
                    next_state = None

                # Store the transition in memory
                self.dqn_agent.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                self.dqn_agent.optimize_model()

                if done:
                    self.episode_durations.append(t + 1)
                    self.plot_durations()
                    break
            # Update the target network, copying all weights and biases in DQN
            if i_episode % TARGET_UPDATE == 0:
                self.dqn_agent.update_target_net()

        print('Complete')
        self.env.render()
        self.env.close()
        plt.ioff()
        plt.show()

    def get_screen(self):
        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        # such as 800x1200x3. Transpose it into torch order (CHW).
        screen = self.env.render(mode='rgb_array').transpose((2, 0, 1))
        # Cart is in the lower half, so strip off the top and bottom of the screen
        _, screen_height, screen_width = screen.shape
        screen = screen[:, int(screen_height * 0.4):int(screen_height * 0.8)]
        view_width = int(screen_width * 0.6)
        cart_location = self.get_cart_location(screen_width)
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2,
                                cart_location + view_width // 2)
        # Strip off the edges, so that we have a square image centered on a cart
        screen = screen[:, :, slice_range]
        # Convert to float, rescale, convert to torch tensor
        # (this doesn't require a copy)
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        return resize(screen).unsqueeze(0).to(self.device)

    def get_cart_location(self, screen_width):
        world_width = self.env.x_threshold * 2
        scale = screen_width / world_width
        return int(self.env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

    def plot_durations(self):
        plt.figure(2)
        plt.clf()
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)

