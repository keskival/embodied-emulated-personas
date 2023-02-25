## Usage

```
pip install -r requirements.txt
```

Put your OpenAI API key from https://platform.openai.com/account/api-keys and the organization id from https://platform.openai.com/account/org-settings
to a file named `apikey.json` containing following:
```
{
    "apikey": "yourapikeyfromopenai",
    "org": "yourorganizationid",
    "model": "text-ada-001"
}
```

## Output

With `davinci-003` it manages to get the pole to the upright position at least temporarily. Note that the first four controls are manually written prompt examples and don't correspond to the real Gym game:
```
I am Sir Isaac Newton, the person who invented the laws of mechanics.
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
Now, let's do a much longer example, and I will show how to get the pole to the upright position:
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
```
