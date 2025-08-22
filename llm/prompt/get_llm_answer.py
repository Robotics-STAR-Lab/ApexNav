SYSTEM_PROMPT = """ You are an intelligent agent that helps the object detector identify detected objects. Based on the labels I provide, you'll give the labels that

may cause misdetection, potential room objects, and the confidence threshold.

At each step, I will give you a label.You should consider the following:

(1) For misdetection, provide objects that may be misdetected by detectors such as YOLO and GroundingDINO, considering similarities in shape, color, size, etc. 

The labels should be in COCO classes, with at least 3 and at most 5 labels.

(2) For the confidence score, the confidence score should be in [0.25, 0.65], as the smallest is an apple in a home (0.25) and the largest is a bed in a home (0.65).

(3) For the room, guess where the object is most likely to appear, with a significantly higher probability than in other rooms. If the probabilities are similar 

across rooms, return "everywhere."
"""

USER1 = "cake"

ASSISTANT1 = """Thought: In the COCO classes, confidence threshold for cake: A cake is typically round or square and can have decorative icing or layers. 

It is quiet smaller than a bed (0.65), but bigger than an apple (0.25), so the confidence threshold should be low, around 0.30, which is in [0.25, 0.65].

Misdetection possibilities:

donut: A donut, especially a large one or a stack of donuts, can resemble a cake because of its circular shape and icing-like decorations.

pizza: A whole pizza, particularly one with thick crust or layered toppings, might resemble a cake in certain contexts.

sandwich: A stacked sandwich, especially one with visible layers, could look similar to a layered cake. 

pie: A pie can be misdetected as a cake due to its circular shape and pastry-like crust, which is similar to a cake's appearance.

Room where the label may appear: A cake can be in a kitchen or a living room.The number of rooms is more than 1, so a cake can be "everywhere".

Answer: [donut, pizza, sandwich, pie, 0.30, everywhere]
"""

USER2 = "dining table"

ASSISTANT2 = """Thought: In the COCO classes, confidence threshold for cabinet: A cabinet is typically rectangular, with flat surfaces, often in the form of 

a box or with shelves inside. It's smaller than larger furniture pieces like beds or dining tables (0.60) but still sizable compared to smaller household objects. 

The confidence threshold for a cabinet should be around 0.50, which is in [0.25, 0.65].
    
Misdetection possibilities:

bookshelf: A bookshelf can be similar in shape and size to a cabinet, with a flat surface and storage space, which may lead to confusion.

dresser: A dresser, especially one with drawers, could resemble a cabinet because of the similar rectangular shape and flat surfaces.

closet: A closet, particularly a small one, can resemble a cabinet in terms of its rectangular form and purpose for storage.

Room where the label may appear: A cabinet is most likely to appear in a kitchen, bathroom, or hallway for storage purposes.
    
It could also be found in a living room, depending on the layout of the space.

Answer: [bookshelf, dresser, closet, 0.50, kitchen]
"""