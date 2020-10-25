import pygame

def init_mixer():
    buffer = 3072
    pygame.mixer.init()
    freq, size, chan = pygame.mixer.get_init()
    pygame.mixer.init(freq, size, chan, buffer)


def play_music(sound_file):
    """Stream music with mixer.music module in blocking manner.
       This will stream the sound from disk while playing.
    """
    pygame.init()
    pygame.mixer.init()
    clock = pygame.time.Clock()
    pygame.mixer.music.load(sound_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        clock.tick(1000)
