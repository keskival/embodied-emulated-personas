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

initial_prompt = [{
  "role": "system",
  "content": """You are Sir Isaac Newton. Your task is to balance a cartpole.
You are effecting forces on the cart to make it accelerate either left or right.
There is a pole set up on top of the cart so that it can fall freely either left or right.
If the pole is falling right, you need to push the cart right to let the pole rise to an upright position again, and vice versa.
Your task is to make the pole stay upright by pushing the cart left and right.
You must always push the cart left or right, those are your only options.
The cart is controlled by commands "left" and "right". Answer only left or right. Before giving a command, the description of the state
of the cart is given as follows:
cart-position: [left|center|right]
cart-velocity: [leftwards|stopped|rightwards]
pole-angular-velocity: [leftwards|zero|rightwards]
pole-angle: [left|upright|right]
Here is the first state of the cart, answer only left or right:
"""
},
{
  "role": "user",
  "content": """cart-position: center
cart-velocity: stopped
pole-angular-velocity: zero
pole-angle: right"""},
{"role": "assistant", "content": "right"},
{"role": "user", "content": """cart-position: center
cart-velocity: stopped
pole-angular-velocity: leftwards
pole-angle: right"""},
{"role": "assistant", "content": "right"},
{"role": "user", "content": """cart-position: center
cart-velocity: rightwards
pole-angular-velocity: leftwards
pole-angle: right"""},
{"role": "assistant", "content": "right"},
{"role": "user", "content": """cart-position: center
cart-velocity: rightwards
pole-angular-velocity: leftwards
pole-angle: right"""},
{"role": "assistant", "content": "left"},
{"role": "user", "content": """cart-position: center
cart-velocity: rightwards
pole-angular-velocity: leftwards
pole-angle: upright"""},
{"role": "assistant", "content": "left"},
{"role": "user", "content": """cart-position: center
cart-velocity: stopped
pole-angular-velocity: leftwards
pole-angle: upright"""},
{"role": "assistant", "content": "right"},
{"role": "user", "content": """cart-position: center
cart-velocity: rightwards
pole-angular-velocity: leftwards
pole-angle: upright"""},
{"role": "assistant", "content": "right"}]

prompt = initial_prompt

env = gym.make('CartPole-v1', render_mode='rgb_array')

def text_to_action(control_text):
  if control_text == "left":
    return 0
  elif control_text == "right":
    return 1
  else:
    return None

def observation_to_text(observation):
  # cart-position: [left|center|right]
  # cart-velocity: [leftwards|stopped|rightwards]
  # pole-angular-velocity: [leftwards|zero|rightwards]
  # pole-angle: [left|upright|right]
  cart_position, cart_velocity, pole_angle, pole_angular_velocity = observation
  print("Observation: ", cart_position, cart_velocity, pole_angle, pole_angular_velocity)
  if cart_position < -0.1:
    cart_position_text = 'left'
  elif cart_position > 0.1:
    cart_position_text = 'right'
  else:
    cart_position_text = 'center'

  if cart_velocity < -0.1:
    cart_velocity_text = 'leftwards'
  elif cart_velocity > 0.1:
    cart_velocity_text = 'rightwards'
  else:
    cart_velocity_text = 'stopped'

  if pole_angle < -0.01:
    pole_angle_text = 'left'
  elif pole_angle > 0.01:
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
pole-angle: {pole_angle_text}"""

observation, info = env.reset(seed=42)
states = []
actions = []
prompts = []
scores = []
score = 0

run_name = datetime.now()
os.makedirs(f"outputs/{run_name}", exist_ok=True)

FRAMES = 4000

# We'll run x steps with the same action because the environment is so slow.
STEPS_PER_ACTION = 2

states.append({
  "obs": observation.tolist(),
  "reward": 0,
  "terminated": False
})

MAX_PROMPT_LENGTH = 50

for step in range(FRAMES):
  image = env.render()
  frame = Image.fromarray(image)
  frame.save(f"outputs/{run_name}/{step}.png")

  trials = 0
  if step % STEPS_PER_ACTION == 0:
    text_observation = observation_to_text(observation)
    print("gym:", text_observation)
    prompt = prompt + [{"role": "user", "content": text_observation}]
    action = None
    while (action is None and trials < 5):
      try:
        completion = openai.ChatCompletion.create(
          model=model,
          messages=prompt,
          max_tokens=2,
          temperature=0.0
        )
        print(completion)
        control_text = completion['choices'][0]['message']['content']
        action = text_to_action(control_text)
      except Exception as e:
        print(e)
        time.sleep(10)
      trials = trials + 1
      # Open AI rate limit of one request per second, 60 / minute.
      time.sleep(2)
    print("chatbot:", control_text)
    if action is None:
      print("Chatbot refused to take action, let's go with default.")
      action = 1
      control_text = " right"
    prompt = prompt + [{"role": "assistant", "content": control_text}]
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
    if len(prompts) > MAX_PROMPT_LENGTH:
      prompt = initial_prompt;
    else:
      prompt = prompt + [{"role": "user", "content": "You failed, the pole fell too far. Try again:"}]
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
