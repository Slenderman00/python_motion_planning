{
  pkgs,
  ...
}:
{
  languages.python = {
    enable = true;

    venv = {
      enable = true;
      requirements = ./requirements.txt;
    };

    libraries = with pkgs; [
      zlib
      dbus # libdbus-1.so.3
      fontconfig # libfontconfig.so.1
      freetype # libfreetype.so.6
      glib # libglib-2.0.so.0
      libGL # libGL.so.1
      libxkbcommon # libxkbcommon.so.0
      xorg.libX11 # libX11.so.6
      xorg.libXrender
      wayland
    ];
  };

  # See full reference at https://devenv.sh/reference/options/
}
