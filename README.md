# Learning Site-Specific Beam Alignment

Code and result figures for the following paper:

**Site-Specific Beam Alignment without Explicit Channel Knowledge via Deep Learning**  
Jongwoo Kwak&#8224;, Hanju Yoo&#8224;, Ian P. Roberts, and Chan-Byoung Chae  
To appear in *IEEE Transactions on Wireless Communications*, 2026.  
&#8224; Equal contribution.

## Citation

~~~bibtex
@article{kwak2025site,
  author  = {Kwak, Jongwoo and Yoo, Hanju and Roberts, Ian P. and Chae, Chan-Byoung},
  journal = {To appear in IEEE Transactions on Wireless Communications},
  title   = {Site-Specific Beam Alignment without Explicit Channel Knowledge via Deep Learning},
  year    = {2026}
}
~~~

## Code

Main training entry points are <code>train.py</code>, <code>train_dft.py</code>, and <code>train_norandomphase.py</code>. Plotting and analysis scripts are provided at the repository root. Dataset files, saved model checkpoints, and intermediate <code>.mat</code> files are not included; generate or obtain the channel data described by the paper before running experiments. The scripts expect data under <code>Dataset/</code>, which is excluded from version control.

## License and third-party notices

The project is licensed under GNU GPL v3; see [LICENSE](LICENSE). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution and third-party license information.
