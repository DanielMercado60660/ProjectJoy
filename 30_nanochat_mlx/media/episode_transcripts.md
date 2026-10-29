# Grid Courier episodes played by `checkpoints/sft/last` (greedy decoding, KV cache)

## test seed 10000000: SUCCESS in 19 actions

```
task: put the book in the basket, then put the apple in the box
you: (4,4) holding: nothing
items: book (0,2) | apple (4,3) | cup (1,1)
containers: basket (4,2) | box (1,4)
> move west
> move west
> move west
> move west
> move south
> move south
> pick book
> move east
> move east
> move east
> move east
> put book in basket
> move north
> pick apple
> move west
> move west
> move west
> move north
> put apple in box
```

## test seed 10000001: SUCCESS in 22 actions

```
task: put the gem in the box, then put the coin in the chest
you: (4,0) holding: nothing
items: gem (2,3) | coin (1,3) | key (1,0)
containers: box (0,0) | chest (4,4)
> move west
> move west
> move north
> move north
> move north
> pick gem
> move west
> move west
> move south
> move south
> move south
> put gem in box
> move east
> move north
> move north
> move north
> pick coin
> move east
> move east
> move east
> move north
> put coin in chest
```

## test seed 10000002: SUCCESS in 20 actions

```
task: put the apple in the chest, then put the gem in the basket
you: (3,3) holding: nothing
items: apple (4,0) | gem (1,3) | key (0,1)
containers: chest (0,4) | basket (0,2)
> move east
> move south
> move south
> move south
> pick apple
> move west
> move west
> move west
> move west
> move north
> move north
> move north
> move north
> put apple in chest
> move east
> move south
> pick gem
> move west
> move south
> put gem in basket
```

## test seed 10000003: SUCCESS in 18 actions

```
task: put the key in the basket, then put the cup in the box
you: (0,3) holding: nothing
items: key (0,4) | cup (1,4) | gem (2,0)
containers: basket (4,2) | box (0,3)
> move north
> pick key
> move east
> move east
> move east
> move east
> move south
> move south
> put key in basket
> move west
> move west
> move west
> move north
> move north
> pick cup
> move west
> move south
> put cup in box
```

## test seed 10000004: SUCCESS in 15 actions

```
task: put the cup in the box, then put the apple in the basket
you: (3,2) holding: nothing
items: cup (1,3) | apple (3,3) | gem (4,0)
containers: box (2,4) | basket (0,2)
> move west
> move west
> move north
> pick cup
> move east
> move north
> put cup in box
> move east
> move south
> pick apple
> move west
> move west
> move west
> move south
> put apple in basket
```

## test seed 10000005: SUCCESS in 8 actions

```
task: put the gem in the basket
you: (2,2) holding: nothing
items: gem (3,4) | book (0,2) | key (0,0)
containers: basket (2,2) | chest (0,4)
> move east
> move north
> move north
> pick gem
> move west
> move south
> move south
> put gem in basket
```

## test3 seed 30000000: FAIL in 40 actions

```
task: put the cup in the box, then put the key in the basket, then put the gem in the box
you: (0,2) holding: nothing
items: cup (0,0) | key (1,2) | gem (4,1)
containers: box (2,1) | basket (3,3)
> move south
> move south
> pick cup
> move east
> move east
> move north
> put cup in box
> move west
> move north
> pick key
> move east
> move east
> move north
> put key in box    [INVALID]
> put key in box    [INVALID]
> put key in box    [INVALID]
> put key in box    [INVALID]
> put key in box    [INVALID]
> put key in box    [INVALID]
> put key in box    [INVALID]
> put key in box    [INVALID]
> put key in box    [INVALID]
> put key in box    [INVALID]
> put key in box    [INVALID]
> put key in box    [INVALID]
> put key in box    [INVALID]
> put key in box    [INVALID]
> put key in box    [INVALID]
> put key in box    [INVALID]
> put key in box    [INVALID]
> put key in box    [INVALID]
> put key in box    [INVALID]
> put key in box    [INVALID]
> put key in box    [INVALID]
> put key in box    [INVALID]
> put key in box    [INVALID]
> put key in box    [INVALID]
> put key in box    [INVALID]
> put key in box    [INVALID]
> put key in box    [INVALID]
```

## test3 seed 30000001: FAIL in 40 actions

```
task: put the coin in the chest, then put the gem in the chest, then put the apple in the basket
you: (1,0) holding: nothing
items: coin (1,2) | gem (1,0) | apple (3,1)
containers: chest (0,4) | basket (2,0)
> move north
> move north
> pick coin
> move west
> move north
> move north
> put coin in chest
> move east
> move south
> move south
> move south
> move south
> pick gem
> move east
> put gem in basket
> move east
> move north
> pick gem    [INVALID]
> pick gem    [INVALID]
> pick gem    [INVALID]
> pick gem    [INVALID]
> pick gem    [INVALID]
> pick gem    [INVALID]
> pick gem    [INVALID]
> pick gem    [INVALID]
> pick gem    [INVALID]
> pick gem    [INVALID]
> pick gem    [INVALID]
> pick gem    [INVALID]
> pick gem    [INVALID]
> pick gem    [INVALID]
> pick gem    [INVALID]
> pick gem    [INVALID]
> pick gem    [INVALID]
> pick gem    [INVALID]
> pick gem    [INVALID]
> pick gem    [INVALID]
> pick gem    [INVALID]
> pick gem    [INVALID]
> pick gem    [INVALID]
```

## test3 seed 30000002: FAIL in 40 actions

```
task: put the apple in the box, then put the gem in the box, then put the book in the basket
you: (0,0) holding: nothing
items: apple (4,2) | gem (0,0) | book (0,1)
containers: box (0,2) | basket (3,4)
> move east
> move east
> move east
> move east
> move north
> move north
> pick apple
> move west
> move west
> move west
> move west
> put apple in box
> move south
> move south
> pick gem
> move east
> move east
> move east
> move north
> move north
> move north
> move north
> put gem in basket
> move west
> move west
> move west
> move south
> move south
> move south
> pick gem    [INVALID]
> pick gem    [INVALID]
> pick gem    [INVALID]
> pick gem    [INVALID]
> pick gem    [INVALID]
> pick gem    [INVALID]
> pick gem    [INVALID]
> pick gem    [INVALID]
> pick gem    [INVALID]
> pick gem    [INVALID]
> pick gem    [INVALID]
```

## test3 seed 30000003: FAIL in 40 actions

```
task: put the apple in the box, then put the cup in the basket, then put the book in the basket
you: (4,1) holding: nothing
items: apple (3,2) | cup (1,0) | book (1,4)
containers: box (3,1) | basket (2,3)
> move west
> move north
> pick apple
> move south
> put apple in box
> move west
> move west
> move south
> pick cup
> move east
> move north
> move north
> move north
> put cup in basket
> move west
> move north
> pick cup    [INVALID]
> pick cup    [INVALID]
> pick cup    [INVALID]
> pick cup    [INVALID]
> pick cup    [INVALID]
> pick cup    [INVALID]
> pick cup    [INVALID]
> pick cup    [INVALID]
> pick cup    [INVALID]
> pick cup    [INVALID]
> pick cup    [INVALID]
> pick cup    [INVALID]
> pick cup    [INVALID]
> pick cup    [INVALID]
> pick cup    [INVALID]
> pick cup    [INVALID]
> pick cup    [INVALID]
> pick cup    [INVALID]
> pick cup    [INVALID]
> pick cup    [INVALID]
> pick cup    [INVALID]
> pick cup    [INVALID]
> pick cup    [INVALID]
> pick cup    [INVALID]
```

## test3 seed 30000004: FAIL in 40 actions

```
task: put the apple in the basket, then put the book in the chest, then put the coin in the basket
you: (4,2) holding: nothing
items: apple (0,4) | book (4,2) | coin (3,0)
containers: basket (3,2) | chest (4,0)
> move west
> move west
> move west
> move west
> move north
> move north
> pick apple
> move east
> move east
> move east
> move south
> move south
> put apple in basket
> move east
> pick book
> move west
> put book in basket
> move south
> move south
> pick book    [INVALID]
> pick book    [INVALID]
> pick book    [INVALID]
> pick book    [INVALID]
> pick book    [INVALID]
> pick book    [INVALID]
> pick book    [INVALID]
> pick book    [INVALID]
> pick book    [INVALID]
> pick book    [INVALID]
> pick book    [INVALID]
> pick book    [INVALID]
> pick book    [INVALID]
> pick book    [INVALID]
> pick book    [INVALID]
> pick book    [INVALID]
> pick book    [INVALID]
> pick book    [INVALID]
> pick book    [INVALID]
> pick book    [INVALID]
> pick book    [INVALID]
```

## test3 seed 30000005: FAIL in 40 actions

```
task: put the coin in the box, then put the apple in the box, then put the gem in the basket
you: (4,4) holding: nothing
items: coin (4,1) | apple (2,0) | gem (0,2)
containers: box (1,3) | basket (2,3)
> move south
> move south
> move south
> pick coin
> move west
> move west
> move west
> move north
> move north
> put coin in box
> move east
> move south
> move south
> move south
> pick apple
> move north
> move north
> move north
> put apple in basket
> move west
> move west
> move south
> pick apple    [INVALID]
> pick apple    [INVALID]
> pick apple    [INVALID]
> pick apple    [INVALID]
> pick apple    [INVALID]
> pick apple    [INVALID]
> pick apple    [INVALID]
> pick apple    [INVALID]
> pick apple    [INVALID]
> pick apple    [INVALID]
> pick apple    [INVALID]
> pick apple    [INVALID]
> pick apple    [INVALID]
> pick apple    [INVALID]
> pick apple    [INVALID]
> pick apple    [INVALID]
> pick apple    [INVALID]
> pick apple    [INVALID]
```

## test_seen seed 20000000: SUCCESS in 7 actions

```
task: put the gem in the chest
you: (0,4) holding: nothing
items: gem (0,3) | cup (0,4) | book (3,1)
containers: chest (3,4) | basket (4,3)
> move south
> pick gem
> move east
> move east
> move east
> move north
> put gem in chest
```

## test_seen seed 20000001: SUCCESS in 17 actions

```
task: put the coin in the basket, then put the gem in the chest
you: (4,4) holding: nothing
items: coin (1,3) | gem (3,4) | key (1,1)
containers: basket (4,4) | chest (4,1)
> move west
> move west
> move west
> move south
> pick coin
> move east
> move east
> move east
> move north
> put coin in basket
> move west
> pick gem
> move east
> move south
> move south
> move south
> put gem in chest
```

## test_seen seed 20000002: SUCCESS in 16 actions

```
task: put the key in the box, then put the apple in the basket
you: (2,0) holding: nothing
items: key (0,4) | apple (4,4) | cup (3,4)
containers: box (1,4) | basket (2,4)
> move west
> move west
> move north
> move north
> move north
> move north
> pick key
> move east
> put key in box
> move east
> move east
> move east
> pick apple
> move west
> move west
> put apple in basket
```

## test_seen seed 20000003: SUCCESS in 21 actions

```
task: put the gem in the chest, then put the book in the chest
you: (1,1) holding: nothing
items: gem (0,0) | book (3,1) | cup (4,1)
containers: chest (1,4) | basket (2,3)
> move west
> move south
> pick gem
> move east
> move north
> move north
> move north
> move north
> put gem in chest
> move east
> move east
> move south
> move south
> move south
> pick book
> move west
> move west
> move north
> move north
> move north
> put book in chest
```

## test_seen seed 20000004: SUCCESS in 20 actions

```
task: put the cup in the basket, then put the key in the box
you: (2,3) holding: nothing
items: cup (3,1) | key (0,3) | book (2,1)
containers: basket (1,4) | box (4,1)
> move east
> move south
> move south
> pick cup
> move west
> move west
> move north
> move north
> move north
> put cup in basket
> move west
> move south
> pick key
> move east
> move east
> move east
> move east
> move south
> move south
> put key in box
```

## test_seen seed 20000005: SUCCESS in 11 actions

```
task: put the apple in the box, then put the key in the box
you: (1,2) holding: nothing
items: apple (1,3) | key (2,1) | book (3,3)
containers: box (3,1) | chest (1,4)
> move north
> pick apple
> move east
> move east
> move south
> move south
> put apple in box
> move west
> pick key
> move east
> put key in box
```

