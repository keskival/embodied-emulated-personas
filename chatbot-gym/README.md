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

[The animated gif](./newton-zeroshot-cartpole.gif).

Example result for one episode:
```
Allow me to introduce myself: Sir Isaac Newton,
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
cart-position: center
cart-velocity: rightwards
pole-angular-velocity: leftwards
pole-angle: left
push-cart: right
cart-position: center
cart-velocity: rightwards
pole-angular-velocity: leftwards
pole-angle: left
push-cart: left
cart-position: center
cart-velocity: rightwards
pole-angular-velocity: leftwards
pole-angle: left
push-cart: left
cart-position: center
cart-velocity: stopped
pole-angular-velocity: rightwards
pole-angle: left
push-cart: left
cart-position: center
cart-velocity: leftwards
pole-angular-velocity: rightwards
pole-angle: left
push-cart: right
cart-position: center
cart-velocity: stopped
pole-angular-velocity: zero
pole-angle: left
push-cart: right
cart-position: center
cart-velocity: rightwards
pole-angular-velocity: leftwards
pole-angle: left
push-cart: right
cart-position: center
cart-velocity: rightwards
pole-angular-velocity: leftwards
pole-angle: left
push-cart: left
cart-position: center
cart-velocity: rightwards
pole-angular-velocity: leftwards
pole-angle: left
push-cart: left
cart-position: center
cart-velocity: stopped
pole-angular-velocity: zero
pole-angle: left
push-cart: left
cart-position: center
cart-velocity: leftwards
pole-angular-velocity: rightwards
pole-angle: left
push-cart: right
cart-position: center
cart-velocity: stopped
pole-angular-velocity: leftwards
pole-angle: left
push-cart: right
cart-position: center
cart-velocity: rightwards
pole-angular-velocity: leftwards
pole-angle: left
push-cart: right
cart-position: center
cart-velocity: rightwards
pole-angular-velocity: leftwards
pole-angle: far-left
push-cart: right
```
