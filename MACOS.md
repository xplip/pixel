

On Mac OS 13+, you need to explicitly prevent `fontconfig` from using the default system fonts. Setting this up requires five steps:

1. Install the fallback fonts

```bash
python scripts/data/download_fallback_fonts.py <output_dir>
```

2. Install fontconfig using [Homebrew](https://brew.sh/)
```bash
brew install fontconfig
```

3. Create a PIXEL-specific font configuration file
```
cd /opt/homebrew/etc/fonts
cp fonts.conf pixel.conf
```

4. Edit the PIXEL-specific configuration file `pixel.conf`

Near the top of the file, you should see the following block
```
<!-- Font directory list -->

        <dir>/System/Library/Fonts</dir> <dir>/Library/Fonts</dir> <dir>~/Library/Fonts</dir> <dir>/System/Library/Assets/com_apple_MobileAsset_Font3</dir> <dir>/System/Library/Assets/com_apple_MobileAsset_Font4</dir>
        <dir>/System/Library/Fonts</dir> <dir>/Library/Fonts</dir> <dir>~/Library/Fonts</dir> <dir>/System/Library/AssetsV2/com_apple_MobileAsset_Font7</dir>
        <dir prefix="xdg">fonts</dir>
        <!-- the following element will be removed in the future -->
        <dir>~/.fonts</dir>

<!--
  Accept deprecated 'mono' alias, replacing it with 'monospace'
-->
```
Turn it into this
```
<!-- Font directory list -->

        <dir>/path/to/your/fallback/fonts/from/step/1/</dir>

<!--
  Accept deprecated 'mono' alias, replacing it with 'monospace'
-->
```

5. Edit the rendering script src/pixel/data/rendering/pangocairo_renderer.py

Add the following two lines to the end of the imports:

```python
os.environ["PANGOCAIRO_BACKEND"] = "fontconfig"
os.environ["FONTCONFIG_FILE"] = "/opt/homebrew/etc/fonts/pixel.conf"
```

FAQ

1. Does this also apply to other versions of Mac OS?
 * Not sure. Let us know if it does!

2. I don't have the directory `/opt/homebrew/etc/fonts`
 * You may and an older/custom installation of Homebrew that placed everything in `/usr/local/homebrew/etc/fonts/`. Run `which brew` to see where homebrew is installed.
