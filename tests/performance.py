# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 21:00:24 2023

@author: mbelic
"""
import cProfile
import pstats
from languageModel import LanguageProcessingModel

def performance_test():
    model = LanguageProcessingModel()
    input_text = "Sample text for testing"
    model.comprehension(input_text)

if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()
    
    performance_test()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.sort_stats('time')
    stats.print_stats(10)
    
    stats.sort_stats('call')
    stats.print_stats(10)
