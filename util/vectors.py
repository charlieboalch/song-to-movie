class MediaVector:
    def __init__(self, valence, energy, darkness, tension, warmth, humor):
        self.valence = valence
        self.energy = energy
        self.darkness = darkness
        self.tension = tension
        self.warmth = warmth
        self.humor = humor

    def adjust_rankings(self, array):
        self.valence += array[0]
        self.energy += array[1]
        self.darkness += array[2]
        self.tension += array[3]
        self.warmth += array[4]
        self.humor += array[5]

    def to_list(self):
        return [self.valence, self.energy, self.darkness, self.tension, self.warmth, self.humor]
