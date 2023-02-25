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
Pray, permit me to introduce myself, Sir Isaac Newton, the discoverer of the laws of mechanics.
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
cart-position: center
cart-velocity: stopped
pole-angular-velocity: zero
pole-angle: upright
push-cart: right
cart-position: center
cart-velocity: rightwards
pole-angular-velocity: leftwards
pole-angle: upright
push-cart: left
cart-position: center
cart-velocity: stopped
pole-angular-velocity: zero
pole-angle: left
push-cart: right
cart-position: center
cart-velocity: rightwards
pole-angular-velocity: leftwards
pole-angle: left
push-cart: left
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
cart-position: right
cart-velocity: rightwards
pole-angular-velocity: leftwards
pole-angle: left
push-cart: left
cart-position: right
cart-velocity: stopped
pole-angular-velocity: leftwards
pole-angle: far-left
push-cart: right
cart-position: right
cart-velocity: rightwards
pole-angular-velocity: leftwards
pole-angle: far-left
push-cart: right
```
