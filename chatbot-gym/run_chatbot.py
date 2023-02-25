#!/usr/bin/python

import gymnasium as gym
import openai
import json

with open('apikey.json', 'r') as apikey_file:
  config = json.load(apikey_file)
  openai.api_key  = config['apikey']
  openai.organization = config['org']
  model = config['model']

prompt = """I am Sir Isaac Newton, the person who invented the laws of mechanics.
Let me demonstrate how to balance a text-controlled cartpole.
The cartpole consists of a freely moving cart which can move left and right, and a pole balanced on top of it.
The aim is to keep the pole upright while keeping the cart from hitting the limits of the track.
In other words, a pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces in the left
and right direction on the cart.
The pole angle is the most important variable to control, then the pole angular velocity.
The state of the cart is described like follows:
cart-position: [left-limit|left|center|right|right-limit]
cart-velocity: [left|stopped|right]
pole-angle: [far-left|left|upright|right|far-right]
pole-angular-velocity: [left|zero|right]
The control of the cart is described as follows:
control: [left|right]
I will iterate states and controls in sequence to show how to balance the cart. Let's start:
cart-position: left
cart-velocity: stopped
pole-angle: right
pole-angular-velocity: zero
control: right
cart-position: center
cart-velocity: right
pole-angle: right
pole-angular-velocity: left
control: right
cart-position: center
cart-velocity: right
pole-angle: upright
pole-angular-velocity: left
control: left
cart-position: right
cart-velocity: stopped
pole-angle: upright
pole-angular-velocity: zero
control: left
"""

env = gym.make('CartPole-v1')

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

for steps in range(10):

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
  prompt = prompt + control_text + "\n"

  observation, reward, terminated, truncated, info = env.step(action)
  if terminated or truncated:
    print("Episode ended.")
    observation, info = env.reset()
print("The episode unfolded as follows:")
print(prompt)
env.close()
