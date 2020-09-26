#!/bin/zsh

rsync -arv --exclude="tmp*" rusty:"~/projects/chemical-torus-imaging/cache/" ~/projects/chemical-torus-imaging/cache
