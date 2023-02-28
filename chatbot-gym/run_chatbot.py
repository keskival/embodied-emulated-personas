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

initial_prompt = """Allow me to introduce myself: Sir Isaac Newton,
the discoverer of the laws of mechanics.
I humbly present an exhibition on the art of equilibrating a text-managed cartpole.
This apparatus consists of a carriage that moves freely to and fro,
with an elongated pole perched atop that necessitates balancing.
The task at hand is to maintain the pole's vertical orientation by exerting forces
to the left when it falls to the left and to the right when it falls to the right.
In essence, the joint's powerlessness impels the pole to cling to a carriage that
travels along an unresisting pathway. The pendulum is situated uprightly
on the carriage, and balance is maintained by applying forces in the leftward
and rightward directions upon the carriage. The angle of the pole is
the primary quantity to control, followed by the angular velocity of the pole.
The state of the carriage is described in the subsequent manner:
cart-position: [left-limit|left|center|right|right-limit]
cart-velocity: [leftwards|stopped|rightwards]
pole-angular-velocity: [leftwards|zero|rightwards]
pole-angle: [far-left|left|upright|right|far-right]
The control of the cart is described as follows:
push-cart: [left|right]
Upon observing the pole's inclination towards the right,
I shall apply a force towards the right on the cart to establish the pole's vertical position,
and conversely for the leftward inclination. With due respect,
I shall sequentially iterate through the states and controls to demonstrate
the art of stabilizing the pole angle upright. Let us commence forthwith:
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
Pray, allow me to present a more extensive example,
wherein I shall illustrate the maneuvers required to position the pole to an upright and vertical angle:
"""

prompt = initial_prompt

env = gym.make('CartPole-v1', render_mode='rgb_array')

def text_to_action(control_text):
  if control_text == " left":
    return 0
  elif control_text == " right":
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
push-cart:"""

observation, info = env.reset(seed=42)
states = []
actions = []
prompts = []
scores = []
score = 0

run_name = datetime.now()
os.makedirs(f"outputs/{run_name}", exist_ok=True)

FRAMES = 1000

# We'll run x steps with the same action because the environment is so slow.
STEPS_PER_ACTION = 2

states.append({
  "obs": observation.tolist(),
  "reward": 0,
  "terminated": False
})

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
          max_tokens=1,
          temperature=0.01
        )
        print(completion)
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
      control_text = " right"
    prompt = prompt + control_text + "\n"
  actions.append(action)

  observation, reward, terminated, truncated, info = env.step(action)

  states.append({
    "obs": observation.tolist(),
    "reward": reward,
    "terminated": terminated
  })
  score = score + reward

  if terminated or truncated:
    scores.append(score)
    score = 0
    print("Episode ended.")
    observation, info = env.reset()
    prompts.append(prompt)
    prompt = initial_prompt;
    states.append({
      "obs": observation.tolist(),
      "reward": 0,
      "terminated": False
    })
  with open(f"outputs/{run_name}/sequence.json", "w") as sequence:
    sequence.write(json.dumps(list(zip(states, actions))))
  with open(f"outputs/{run_name}/scores.json", "w") as sequence:
    sequence.write(json.dumps(scores))
  with open(f"outputs/{run_name}/prompts.json", "w") as promptsfile:
    promptsfile.write(json.dumps(prompts))
prompts.append(prompt)
print("The episodes unfolded as follows:")
print(prompts)
print("Got scores: ", scores)
env.close()
