; Inno Setup 6 script for Glitch Lab (per-user or all-users install, x64 only).
#define AppName "Glitch Lab"
#define AppVersion "1.0.0"

[Setup]
AppId={{5B7C2E7A-3D3B-4C5E-9F43-6A1E2C7D9B10}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=Glitch Lab
DefaultDirName={autopf}\Glitch Lab
DefaultGroupName=Glitch Lab
OutputDir=..\dist
OutputBaseFilename=GlitchLab-{#AppVersion}-setup-win64
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequiredOverridesAllowed=dialog
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
SetupIconFile=..\src\glitchlab\resources\glitchlab.ico
UninstallDisplayIcon={app}\GlitchLab.exe
LicenseFile=..\LICENSE
MinVersion=10.0

[Files]
Source: "..\dist\GlitchLab\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{group}\Glitch Lab"; Filename: "{app}\GlitchLab.exe"
Name: "{group}\Glitch Lab User Guide"; Filename: "{app}\docs\USER_GUIDE.md"
Name: "{group}\Uninstall Glitch Lab"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Glitch Lab"; Filename: "{app}\GlitchLab.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Registry]
Root: HKA; Subkey: "Software\Classes\.glab"; ValueType: string; ValueName: ""; ValueData: "GlitchLab.Project"; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\GlitchLab.Project"; ValueType: string; ValueName: ""; ValueData: "Glitch Lab project"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\GlitchLab.Project\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\GlitchLab.exe,0"
Root: HKA; Subkey: "Software\Classes\GlitchLab.Project\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\GlitchLab.exe"" ""%1"""

[Run]
Filename: "{app}\GlitchLab.exe"; Description: "Launch Glitch Lab"; Flags: nowait postinstall skipifsilent
