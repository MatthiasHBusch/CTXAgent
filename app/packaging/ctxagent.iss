; Inno Setup script for CTXAgent: per-user install, no admin rights.
;
;   iscc packaging\ctxagent.iss                       (version read from ctxagent\version.py)
;   iscc /DMyAppVersion=0.1.0 packaging\ctxagent.iss  (what packaging\build.ps1 passes)
;
; Installs dist\CTXAgent (the PyInstaller onedir bundle) to
; %LOCALAPPDATA%\Programs\CTXAgent. User data lives in %LOCALAPPDATA%\CTXAgent and
; survives uninstall unless "Also remove my data" is ticked (silent uninstall:
; pass /REMOVEDATA=1).
;
; The app was called PiAgent before. Same AppId, so this setup upgrades a PiAgent
; install: it moves to the new folder, the old program folder and shortcuts are
; removed, and the app itself moves %LOCALAPPDATA%\PiAgent (chats, settings) to
; %LOCALAPPDATA%\CTXAgent on its first start (ctxagent/paths.py).

#define MyAppName "CTXAgent"
#define MyAppExe "CTXAgent.exe"
#define OldAppName "PiAgent"
#define MyAppPublisher "Matthias Busch"
; The public repository (installer "support" link; the source lives elsewhere).
#define MyAppURL "https://github.com/MatthiasHBusch/CTXAgent"

#ifndef MyAppVersion
  ; Fallback: parse __version__ = "x.y.z" out of ctxagent\version.py.
  #define VersionFile FileOpen(AddBackslash(SourcePath) + "..\ctxagent\version.py")
  #if VersionFile == 0
    #error Cannot open ctxagent\version.py
  #endif
  #define MyAppVersion ""
  #define VersionLines 0
  #sub ReadVersionLine
    #define public VersionLine FileRead(VersionFile)
    #define public VersionLines VersionLines + 1
    #define public VersionAt Pos('__version__ = "', VersionLine)
    #if VersionAt > 0
      #define public MyAppVersion Copy(VersionLine, VersionAt + 15, Len(VersionLine) - VersionAt - 15)
    #endif
  #endsub
  #for {VersionLines = 0; (MyAppVersion == "") && (FileEof(VersionFile) == 0); VersionLines} ReadVersionLine
  #expr FileClose(VersionFile)
  #if MyAppVersion == ""
    #error Could not read __version__ from ctxagent\version.py
  #endif
#endif

[Setup]
AppId={{A3AD4755-7CB8-4250-B87A-6DAA259BFDF4}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
VersionInfoVersion={#MyAppVersion}
VersionInfoProductName={#MyAppName}
; {userpf} = %LOCALAPPDATA%\Programs
DefaultDirName={userpf}\{#MyAppName}
; Not the PiAgent folder an upgrade would otherwise reuse.
UsePreviousAppDir=no
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
DisableDirPage=auto
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist\installer
OutputBaseFilename={#MyAppName}-Setup-{#MyAppVersion}
SetupIconFile=ctxagent.ico
UninstallDisplayIcon={app}\{#MyAppExe}
UninstallDisplayName={#MyAppName}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; The app holds this mutex while running: setup and uninstall ask to close it.
AppMutex=CTXAgent.Running,PiAgent.Running
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\dist\CTXAgent\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExe}"; WorkingDir: "{app}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExe}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExe}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent

[InstallDelete]
; What the app installed under its former name (PiAgent). Its data folder is moved
; by the app, not here.
Type: filesandordirs; Name: "{userpf}\{#OldAppName}"
Type: files; Name: "{autoprograms}\{#OldAppName}.lnk"
Type: files; Name: "{autodesktop}\{#OldAppName}.lnk"

[UninstallDelete]
; files the app may write next to itself (none expected), then the folder
Type: filesandordirs; Name: "{app}\_internal\__pycache__"
Type: dirifempty; Name: "{app}"

[Code]
var
  RemoveData: Boolean;

function UserDataDir(): String;
begin
  Result := ExpandConstant('{localappdata}\{#MyAppName}');
end;

function InitializeUninstall(): Boolean;
var
  Form: TSetupForm;
  Info: TNewStaticText;
  Box: TNewCheckBox;
  OkButton, CancelButton: TNewButton;
  W: Integer;
begin
  Result := True;
  RemoveData := ExpandConstant('{param:REMOVEDATA|0}') = '1';
  if UninstallSilent() or not DirExists(UserDataDir()) then
    Exit;
  Form := CreateCustomForm(ScaleX(440), ScaleY(150), False, True);
  try
    Form.Caption := 'Uninstall {#MyAppName}';
    Info := TNewStaticText.Create(Form);
    Info.Parent := Form;
    Info.Left := ScaleX(12);
    Info.Top := ScaleY(12);
    Info.Width := Form.ClientWidth - ScaleX(24);
    Info.AutoSize := False;
    Info.WordWrap := True;
    Info.Height := ScaleY(48);
    Info.Caption := '{#MyAppName} will be removed from this computer. Your chats, settings and logs in '
      + UserDataDir() + ' are kept unless you tick the box below.';
    Box := TNewCheckBox.Create(Form);
    Box.Parent := Form;
    Box.Left := ScaleX(12);
    Box.Top := ScaleY(68);
    Box.Width := Form.ClientWidth - ScaleX(24);
    Box.Height := ScaleY(20);
    Box.Caption := 'Also remove my data (chats, settings, logs). This cannot be undone.';
    Box.Checked := RemoveData;

    OkButton := TNewButton.Create(Form);
    OkButton.Parent := Form;
    OkButton.Caption := 'Uninstall';
    OkButton.ModalResult := mrOk;
    OkButton.Default := True;
    CancelButton := TNewButton.Create(Form);
    CancelButton.Parent := Form;
    CancelButton.Caption := 'Cancel';
    CancelButton.ModalResult := mrCancel;
    CancelButton.Cancel := True;
    W := Form.CalculateButtonWidth([OkButton.Caption, CancelButton.Caption]);
    OkButton.Width := W;
    CancelButton.Width := W;
    OkButton.Height := ScaleY(23);
    CancelButton.Height := ScaleY(23);
    CancelButton.Left := Form.ClientWidth - ScaleX(12) - W;
    OkButton.Left := CancelButton.Left - ScaleX(8) - W;
    OkButton.Top := Form.ClientHeight - ScaleY(23 + 12);
    CancelButton.Top := OkButton.Top;
    Form.ActiveControl := CancelButton;

    if Form.ShowModal() = mrOk then
      RemoveData := Box.Checked
    else
      Result := False;
  finally
    Form.Free();
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Code: Integer;
begin
  { Keys live in Windows Credential Manager, outside the data folder: remove them
    while the program files (and so the CLI) still exist. }
  if (CurUninstallStep = usUninstall) and RemoveData then
  begin
    if not Exec(ExpandConstant('{app}\CTXAgentCLI.exe'), '--run-module ctxagent.keys --forget-all',
                '', SW_HIDE, ewWaitUntilTerminated, Code) or (Code <> 0) then
      Log('Could not remove stored API keys (exit ' + IntToStr(Code) + ')');
  end;
  if (CurUninstallStep = usPostUninstall) and RemoveData and DirExists(UserDataDir()) then
  begin
    if not DelTree(UserDataDir(), True, True, True) then
      Log('Could not remove ' + UserDataDir());
  end;
end;
