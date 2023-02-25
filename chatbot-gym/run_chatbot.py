#!/usr/bin/python

import gymnasium as gym
import openai
import json
import os
from datetime import datetime
from PIL import Image

with open('apikey.json', 'r') as apikey_file:
  config = json.load(apikey_file)
  openai.api_key  = config['apikey']
  openai.organization = config['org']
  model = config['model']

prompt = """Pray, permit me to introduce myself, Sir Isaac Newton, the discoverer of the laws of mechanics.
Allow me to present an exhibition on the art of equilibrating a text-managed cartpole.
The aforementioned pole is comprised of a carriage that oscillates freely to and fro, and an elongated rod perched atop it that requires balancing.
The objective is to keep the rod vertically oriented, while simultaneously prohibiting the carriage from colliding with the boundaries of the track.
In other words, a joint without power impels the rod to cling to a carriage, which travels along an unresisting pathway.
The pendulum is situated uprightly on the carriage, and the aim is to maintain balance by exerting forces in the leftward and rightward directions upon the carriage.
The angle of the rod is the primary quantity to control, followed by the angular velocity of the rod.
The condition of the carriage is delineated in the subsequent manner:
cart-position: [left-limit|left|center|right|right-limit]
cart-velocity: [left|stopped|right]
pole-angle: [far-left|left|upright|right|far-right]
pole-angular-velocity: [left|zero|right]
The control of the cart is described as follows:
control: [left|right]
Verily, I shall iterate through the states and controls sequentially to demonstrate the art of stabilizing the cart. Let us commence forthwith:
cart-position: center
cart-velocity: stopped
pole-angle: right
pole-angular-velocity: zero
control: right
cart-position: center
cart-velocity: stopped
pole-angle: right
pole-angular-velocity: left
control: right
cart-position: center
cart-velocity: right
pole-angle: right
pole-angular-velocity: left
control: right
cart-position: center
cart-velocity: right
pole-angle: right
pole-angular-velocity: left
control: left
cart-position: center
cart-velocity: right
pole-angle: upright
pole-angular-velocity: left
control: left
cart-position: center
cart-velocity: stopped
pole-angle: upright
pole-angular-velocity: left
control: right
cart-position: center
cart-velocity: right
pole-angle: upright
pole-angular-velocity: left
control: left
cart-position: center
cart-velocity: stopped
pole-angle: left
pole-angular-velocity: left
control: right
cart-position: center
cart-velocity: right
pole-angle: left
pole-angular-velocity: left
control: left
cart-position: center
cart-velocity: stopped
pole-angle: left
pole-angular-velocity: left
control: right
Now, let us engage in a more extensive example, wherein I shall illustrate how to maneuver the rod to the erect position:
"""

env = gym.make('CartPole-v1', render_mode='rgb_array')

def text_to_action(control_text):
  if control_text == 'control: left':
    return 0
  elif control_text == 'control: right':
    return 1
  else:
    return None

def observation_to_text(observation):
  # cart-position: [left-limit|left|center|right|right-limit]
  # cart-velocity: [left|stopped|right]
  # pole-angle: [far-left|left|upright|right|far-right]
  # pole-angular-velocity: [left|zero|right]
  cart_position, cart_velocity, pole_angle, pole_angular_velocity = observation
  print("Observation: ", cart_position, cart_velocity, pole_angle, pole_angular_velocity)
  if cart_position < -2.3:
    cart_position_text = 'left-limit'
  elif cart_position > 2.3:
    cart_position_text = 'right-limit'
  elif cart_position < -0.1:
    cart_position_text = 'left'
  elif cart_position > 0.1:
    cart_position_text = 'right'
  else:
    cart_position_text = 'center'

  if cart_velocity < -0.2:
    cart_velocity_text = 'left'
  elif cart_velocity > 0.2:
    cart_velocity_text = 'right'
  else:
    cart_velocity_text = 'stopped'

  if pole_angle < -0.15:
    pole_angle_text = 'far-left'
  elif pole_angle > 0.15:
    pole_angle_text = 'far-right'
  elif pole_angle < -0.02:
    pole_angle_text = 'left'
  elif pole_angle > 0.02:
    pole_angle_text = 'right'
  else:
    pole_angle_text = 'upright'

  if pole_angular_velocity < -0.2:
    pole_angular_velocity_text = 'left'
  elif pole_angular_velocity > 0.2:
    pole_angular_velocity_text = 'right'
  else:
    pole_angular_velocity_text = 'zero'
  return f"""cart-position: {cart_position_text}
cart-velocity: {cart_velocity_text}
pole-angle: {pole_angle_text}
pole-angular-velocity: {pole_angular_velocity_text}
"""

observation, info = env.reset(seed=42)
states = []
actions = []

run_name = datetime.now()
os.makedirs(f"outputs/{run_name}", exist_ok=True)

for step in range(100):
  image = env.render()
  frame = Image.fromarray(image)
  frame.save(f"outputs/{run_name}/{step}.png")

  states.append(observation.tolist())

  trials = 0
  action = None
  text_observation = observation_to_text(observation)
  print("gym: ", text_observation)
  prompt = prompt + text_observation
  while (action is None and trials < 5):
    completion = openai.Completion.create(
      model=model,
      prompt=prompt,
      max_tokens=3,
      temperature=0.01
    )
    control_text = completion['choices'][0]['text']
    action = text_to_action(control_text)
    trials = trials + 1
  print("chatbot: ", control_text)
  if action is None:
    print("Chatbot refused to take action, let's go with default.")
    action = 1
    control_text = 'control: left'
  actions.append(action)
  prompt = prompt + control_text + "\n"

  observation, reward, terminated, truncated, info = env.step(action)
  if terminated or truncated:
    print("Episode ended.")
    observation, info = env.reset()
    prompt = prompt + "\nVery well, allow me to present one further extensive example:\n";
print("The episode unfolded as follows:")
print(prompt)
env.close()

with open(f"outputs/{run_name}/sequence.json", "w") as run:
  run.write(json.dumps(list(zip(states, actions))))
