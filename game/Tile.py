class Tile():
    def __init__(self):
        self.value = ' '
        self.display = 'H'

    def get_value(self):
        return self.value

    def is_hidden(self):
        return self.display == 'H'
    
    def is_flagged(self):
        return self.display == 'F'

    def is_hidden_or_flagged(self):
        return self.display in ('H', 'F')
    
    def is_mine(self):
        return self.value == '*'

    def is_empty(self):
        return self.value == ' '

    def place_mine(self):
        self.value = '*'

    def place_number(self, value):
        self.value = str(value)

    def reveal(self):
        self.display = 'R'

    def toggle_flag(self):
        if self.display == 'H':
            self.display = 'F'
        elif self.display == 'F':
            self.display = 'H'