#!/usr/bin/python

import gymnasium as gym
import openai
import json
import os
from datetime import datetime
from PIL import Image
import time

with open('apikey.json', 'r') as apikey_file:
  config = json.load(apikey_file)
  openai.api_key  = config['apikey']
  openai.organization = config['org']
  model = config['model']

initial_prompt = """Pray, permit me to introduce myself, Sir Isaac Newton, the discoverer of the laws of mechanics.
Allow me to present an exhibition on the art of equilibrating a text-managed cartpole.
The aforementioned pole is comprised of a carriage that oscillates freely to and fro, and an elongated pole perched atop it that requires balancing.
The objective is to keep the pole vertically oriented, while simultaneously prohibiting the carriage from colliding with the boundaries of the track.
In other words, a joint without power impels the pole to cling to a carriage, which travels along an unresisting pathway.
The pendulum is situated uprightly on the carriage, and the aim is to maintain balance by exerting forces in the leftward and rightward directions upon the carriage.
The angle of the pole is the primary quantity to control, followed by the angular velocity of the pole.
The condition of the carriage is delineated in the subsequent manner:
cart-position: [left-limit|left|center|right|right-limit]
cart-velocity: [leftwards|stopped|rightwards]
pole-angular-velocity: [leftwards|zero|rightwards]
pole-angle: [far-left|left|upright|right|far-right]
The control of the cart is described as follows:
push-cart: [left|right]
Whenever the angle of the pole is towards the right, I shall push the cart towards the right to make the pole arise upright, and vice versa.
Verily, I shall iterate through the states and controls sequentially to demonstrate the art of stabilizing the pole angle upright. Let us commence forthwith:
cart-position: center
cart-velocity: stopped
pole-angular-velocity: zero
pole-angle: right
push-cart: right
cart-position: center
cart-velocity: stopped
pole-angular-velocity: leftwards
pole-angle: right
push-cart: right
cart-position: center
cart-velocity: rightwards
pole-angular-velocity: leftwards
pole-angle: right
push-cart: right
cart-position: center
cart-velocity: rightwards
pole-angular-velocity: leftwards
pole-angle: right
push-cart: left
cart-position: center
cart-velocity: rightwards
pole-angular-velocity: leftwards
pole-angle: upright
push-cart: left
cart-position: center
cart-velocity: stopped
pole-angular-velocity: leftwards
pole-angle: upright
push-cart: right
cart-position: center
cart-velocity: rightwards
pole-angular-velocity: leftwards
pole-angle: upright
Now, let us engage in a more extensive example, wherein I shall illustrate how to maneuver the pole to the erect position with an upright angle:
"""

prompt = initial_prompt

env = gym.make('CartPole-v1', render_mode='rgb_array')

def text_to_action(control_text):
  if control_text == 'push-cart: left':
    return 0
  elif control_text == 'push-cart: right':
    return 1
  else:
    return None

def observation_to_text(observation):
  # cart-position: [left-limit|left|center|right|right-limit]
  # cart-velocity: [leftwards|stopped|rightwards]
  # pole-angular-velocity: [leftwards|zero|rightwards]
  # pole-angle: [far-left|left|upright|right|far-right]
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
    cart_velocity_text = 'leftwards'
  elif cart_velocity > 0.2:
    cart_velocity_text = 'rightwards'
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
    pole_angular_velocity_text = 'leftwards'
  elif pole_angular_velocity > 0.2:
    pole_angular_velocity_text = 'rightwards'
  else:
    pole_angular_velocity_text = 'zero'
  return f"""cart-position: {cart_position_text}
cart-velocity: {cart_velocity_text}
pole-angular-velocity: {pole_angular_velocity_text}
pole-angle: {pole_angle_text}
"""

observation, info = env.reset(seed=42)
states = []
actions = []
prompts = []

run_name = datetime.now()
os.makedirs(f"outputs/{run_name}", exist_ok=True)

FRAMES = 1000

# We'll run x steps with the same action because the environment is so slow.
STEPS_PER_ACTION = 2

for step in range(FRAMES):
  image = env.render()
  frame = Image.fromarray(image)
  frame.save(f"outputs/{run_name}/{step}.png")

  states.append(observation.tolist())

  trials = 0
  if step % STEPS_PER_ACTION == 0:
    text_observation = observation_to_text(observation)
    print("gym: ", text_observation)
    prompt = prompt + text_observation
    action = None
    while (action is None and trials < 5):
      try:
        completion = openai.Completion.create(
          model=model,
          prompt=prompt,
          max_tokens=5,
          temperature=0.01
        )
        control_text = completion['choices'][0]['text']
        action = text_to_action(control_text)
      except Exception as e:
        print(e)
        time.sleep(10)
      trials = trials + 1
      # Open AI rate limit of one request per second, 60 / minute.
      time.sleep(2)
    print("chatbot: ", control_text)
    if action is None:
      print("Chatbot refused to take action, let's go with default.")
      action = 1
      control_text = 'push-cart: right'
    prompt = prompt + control_text + "\n"
  actions.append(action)

  observation, reward, terminated, truncated, info = env.step(action)
  if terminated or truncated:
    print("Episode ended.")
    observation, info = env.reset()
    prompts.append(prompt)
    prompt = initial_prompt;
  with open(f"outputs/{run_name}/sequence.json", "w") as sequence:
    sequence.write(json.dumps(list(zip(states, actions))))
  with open(f"outputs/{run_name}/prompts.json", "w") as promptsfile:
    promptsfile.write(json.dumps(prompts))
prompts.append(prompt)
print("The episodes unfolded as follows:")
print(prompts)
env.close()

with open(f"outputs/{run_name}/sequence.json", "w") as sequence:
  sequence.write(json.dumps(list(zip(states, actions))))

with open(f"outputs/{run_name}/prompts.json", "w") as promptsfile:
  promptsfile.write(json.dumps(prompts))
