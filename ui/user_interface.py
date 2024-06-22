import tkinter as tk

class UserInterface:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Universal Consensus')

    def create_widgets(self):
        self.consensus_label = tk.Label(self.root, text='Consensus Value:')
        self.consensus_label.pack()

        self.consensus_value = tk.Entry(self.root)
        self.consensus_value.pack()

        self.submit_button = tk.Button(self.root, text='Submit', command=self.submit_consensus)
        self.submit_button.pack()

    def submit_consensus(self):
        consensus_value = self.consensus_value.get()
        # Call consensus algorithm with consensus value
        pass

    def run(self):
        self.create_widgets()
        self.root.mainloop()
