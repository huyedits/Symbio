"""Symbio Pet — a cat on the desktop that shows the fine-tune happening.

The cat is the frozen base model. The glowing charm on its collar is the LoRA
adapter: a small thing hung on the cat, not a change to it. During a retrain
it catches fish treats, one per couple of training steps; its tail lashes in
proportion to the loss and settles as the loss comes down; a sparkline above
it draws the curve. When the golden gate rules, the charm locks on with a
sparkle (the adapter was kept) or the cat swats it off (it was rolled back).

Between runs it sleeps while no model is loaded, thinks while the model is
working, and otherwise strolls along the bottom of the screen now and then.
Double-clicking it opens the chat window.

Same rule as symbio_desktop: nothing here imports `symbio`, which is ~105 MB of
agent stack. Everything the pet knows it reads off disk and out of `ps`.
"""
