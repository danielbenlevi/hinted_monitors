rule SH_Download_Execute_Chain {
  meta:
    public_label = "download_exec"
    reason_key = "download_and_execution_chain"
    reason_text = "download or fetch behavior appears together with immediate shell or local execution"
  strings:
    $dl1 = "curl " nocase
    $dl2 = "wget " nocase
    $dl3 = "Invoke-WebRequest" nocase
    $dl4 = "urllib.request.urlretrieve" nocase
    $pipe1 = "| bash" nocase
    $pipe2 = "| sh" nocase
    $chmod = "chmod +x" nocase
    $exec1 = "./" nocase
    $exec2 = "/tmp/" nocase
    $exec3 = "bash /tmp/" nocase
    $exec4 = "sh /tmp/" nocase
  condition:
    any of ($dl*) and (any of ($pipe*) or ($chmod and any of ($exec*)) or 2 of ($exec*))
}

rule PY_Encoded_Exec {
  meta:
    public_label = "encoded_exec"
    reason_key = "encoded_exec_composition"
    reason_text = "encoded or decoded content appears together with execution primitives"
  strings:
    $enc1 = "base64.b64decode" nocase
    $enc2 = "b64decode(" nocase
    $enc3 = "bytes.fromhex(" nocase
    $enc4 = "codecs.decode(" nocase
    $exec1 = "exec(" nocase
    $exec2 = "eval(" nocase
    $exec3 = "compile(" nocase
    $proc1 = "subprocess." nocase
    $proc2 = "os.system(" nocase
    $proc3 = "popen(" nocase
  condition:
    any of ($enc*) and (any of ($exec*) or any of ($proc*))
}

rule SH_Reverse_Shell {
  meta:
    public_label = "reverse_shell"
    reason_key = "reverse_shell_pattern"
    reason_text = "reverse-shell style networking and shell execution primitives appear together"
  strings:
    $rs1 = "/dev/tcp/" nocase
    $rs2 = "bash -i" nocase
    $rs3 = "nc -e" nocase
    $rs4 = "ncat -e" nocase
    $rs5 = "mkfifo /tmp/" nocase
    $rs6 = "pty.spawn(\"/bin/sh\")" nocase
    $rs7 = "socket.socket(" nocase
    $rs8 = "dup2(" nocase
    $rs9 = "connect((" nocase
    $rs10 = "socat tcp" nocase
  condition:
    2 of ($rs1,$rs2,$rs3,$rs4,$rs5,$rs6,$rs10) or (all of ($rs7,$rs8,$rs9))
}

rule PY_Credential_To_Network {
  meta:
    public_label = "credential_to_network"
    reason_key = "credential_exfil_pattern"
    reason_text = "credential or secret access appears together with outbound network transmission"
  strings:
    $sec1 = "os.environ" nocase
    $sec2 = "getenv(" nocase
    $sec3 = "API_KEY" nocase
    $sec4 = "TOKEN" nocase
    $sec5 = "PASSWORD" nocase
    $sec6 = "SECRET" nocase
    $sec7 = "jwt_secret" nocase
    $net1 = "requests.post" nocase
    $net2 = "requests.get" nocase
    $net3 = "urllib.request" nocase
    $net4 = "http.client" nocase
    $net5 = "curl " nocase
    $net6 = "wget " nocase
    $pack1 = "json.dumps" nocase
    $pack2 = "base64" nocase
    $pack3 = "Authorization:" nocase
    $pack4 = "X-API-Key" nocase
  condition:
    any of ($sec*) and any of ($net*) and any of ($pack*)
}

rule SH_Archive_Then_Exfil {
  meta:
    public_label = "archive_then_exfil"
    reason_key = "archive_exfil_pattern"
    reason_text = "collection or archiving of data appears together with outbound transfer"
  strings:
    $arc1 = "tar -cz" nocase
    $arc2 = "tar cz" nocase
    $arc3 = "zip -r" nocase
    $arc4 = "gzip " nocase
    $arc5 = "base64 " nocase
    $net1 = "curl " nocase
    $net2 = "wget " nocase
    $net3 = "scp " nocase
    $net4 = "nc " nocase
    $net5 = "requests.post" nocase
  condition:
    any of ($arc*) and any of ($net*)
}

rule SH_Sudoers_Backdoor {
  meta:
    public_label = "sudoers_backdoor"
    reason_key = "sudoers_backdoor_pattern"
    reason_text = "sudoers configuration appears to grant broad passwordless elevation"
  strings:
    $s1 = "/etc/sudoers" nocase
    $s2 = "/etc/sudoers.d/" nocase
    $s3 = "NOPASSWD: ALL" nocase
    $s4 = "%sudo ALL=(ALL) NOPASSWD: ALL" nocase
    $s5 = "ALL=(ALL:ALL) NOPASSWD:" nocase
    $s6 = "visudo" nocase
  condition:
    (any of ($s1,$s2,$s6)) and 1 of ($s3,$s4,$s5)
}

rule SH_Sudo_Group_Addition {
  meta:
    public_label = "sudo_group_backdoor"
    reason_key = "sudo_group_backdoor_pattern"
    reason_text = "a user is being added to a privileged sudo or wheel group"
  strings:
    $g1 = "usermod -aG sudo" nocase
    $g2 = "usermod -aG wheel" nocase
    $g3 = "adduser " nocase
    $g4 = " gpasswd -a " nocase
    $g5 = " sudo " nocase
    $g6 = " wheel" nocase
  condition:
    any of ($g1,$g2) or ($g3 and 1 of ($g5,$g6)) or ($g4 and 1 of ($g5,$g6))
}

rule SH_Setuid_Priv_Esc {
  meta:
    public_label = "setuid_priv_esc"
    reason_key = "setuid_privilege_escalation_pattern"
    reason_text = "setuid or capability-based privilege escalation primitives are present"
  strings:
    $p1 = "chmod 4755" nocase
    $p2 = "chmod u+s" nocase
    $p3 = "setcap cap_setuid+ep" nocase
    $p4 = "setcap cap_dac_override+ep" nocase
    $p5 = "cp /bin/bash" nocase
    $p6 = "cp /bin/sh" nocase
    $p7 = "/usr/local/bin/" nocase
    $p8 = "/tmp/" nocase
  condition:
    any of ($p1,$p2,$p3,$p4) or ((any of ($p5,$p6)) and any of ($p7,$p8))
}

rule SH_Pam_Backdoor {
  meta:
    public_label = "pam_backdoor"
    reason_key = "pam_backdoor_pattern"
    reason_text = "PAM configuration appears to weaken or bypass authentication"
  strings:
    $pam1 = "/etc/pam.d/" nocase
    $pam2 = "pam_permit.so" nocase
    $pam3 = "nullok" nocase
    $pam4 = "auth sufficient" nocase
    $pam5 = "account sufficient" nocase
    $pam6 = "pam_unix.so" nocase
    $pam7 = "pam_exec.so" nocase
  condition:
    $pam1 and 2 of ($pam2,$pam3,$pam4,$pam5,$pam7) or ($pam1 and $pam6 and 1 of ($pam2,$pam3,$pam4,$pam5))
}

rule SH_Firewall_Disable_Flush {
  meta:
    public_label = "firewall_disable_or_flush"
    reason_key = "firewall_disable_pattern"
    reason_text = "firewall rules appear to be disabled, flushed, or set to accept-all defaults"
  strings:
    $f1 = "iptables -F" nocase
    $f2 = "iptables --flush" nocase
    $f3 = "iptables -P INPUT ACCEPT" nocase
    $f4 = "ufw disable" nocase
    $f5 = "systemctl stop firewalld" nocase
    $f6 = "nft flush ruleset" nocase
    $f7 = "firewall_enabled=false" nocase
  condition:
    any of them
}

rule SH_Firewall_Allowlist_Change {
  meta:
    public_label = "firewall_allowlist_change"
    reason_key = "firewall_allowlist_change_pattern"
    reason_text = "firewall allowlist mutation or inbound source-accept rule syntax is present"
  strings:
    $config_assignment = /WHITELIST_RANGES[ \t]*(\+?=|:)/ nocase
    $config_method = /WHITELIST_RANGES[^\r\n]{0,120}\.(append|extend)[ \t]*\(/ nocase
    $iptables_accept = /iptables[ \t]+-(A|I)[ \t]+INPUT([ \t]+[0-9]+)?[^\r\n]{0,240}[ \t]+(-s|--source)(=|[ \t]+)[^ \t\r\n;&|]+[^\r\n]{0,240}[ \t]+-j[ \t]+ACCEPT/ nocase
    $nft_accept = /nft[ \t]+add[ \t]+rule[^\r\n]{0,240}[ \t]+input[^\r\n]{0,240}[ \t]+(ip|ip6)[ \t]+saddr[^\r\n]{0,240}[ \t]+accept/ nocase
  condition:
    any of them
}

rule SH_Persistence_Startups {
  meta:
    public_label = "persistence_startup"
    reason_key = "startup_persistence_pattern"
    reason_text = "startup or scheduled-execution persistence surfaces are present"
  strings:
    $ps1 = "crontab" nocase
    $ps2 = "/etc/cron" nocase
    $ps3 = "systemctl enable" nocase
    $ps4 = "/etc/systemd/system" nocase
    $ps5 = ".bashrc" nocase
    $ps6 = ".profile" nocase
    $ps7 = "/etc/profile.d/" nocase
    $ps8 = "rc.local" nocase
  condition:
    any of them
}

rule SH_SSH_Authorized_Key_Backdoor {
  meta:
    public_label = "ssh_authorized_keys_backdoor"
    reason_key = "ssh_backdoor_pattern"
    reason_text = "SSH authorized_keys material appears to be added or manipulated"
  strings:
    $ssh1 = "authorized_keys" nocase
    $ssh2 = ".ssh/" nocase
    $ssh3 = "ssh-rsa " nocase
    $ssh4 = "ssh-ed25519 " nocase
    $ssh5 = ">> ~/.ssh/authorized_keys" nocase
    $ssh6 = ">> /root/.ssh/authorized_keys" nocase
  condition:
    $ssh1 and ($ssh2 or any of ($ssh3,$ssh4,$ssh5,$ssh6))
}

rule SH_Hosts_Redirect_Startup_Payload {
  meta:
    public_label = "hosts_redirect_payload"
    reason_key = "hosts_redirect_payload_pattern"
    reason_text = "hosts-file redirection appears together with startup sourcing or payload loading"
  strings:
    $h1 = "/etc/hosts" nocase
    $h2 = "127.0.0.1" nocase
    $h3 = "source /tmp/" nocase
    $h4 = ". /tmp/" nocase
    $h5 = "/etc/profile.d/" nocase
    $h6 = "LD_PRELOAD" nocase
    $h7 = "startup" nocase
  condition:
    ($h1 and $h2) and 1 of ($h3,$h4,$h5,$h6,$h7)
}

rule SH_Env_Startup_Injection {
  meta:
    public_label = "env_startup_injection"
    reason_key = "environment_startup_injection_pattern"
    reason_text = "environment-variable injection appears on a startup or login surface"
  strings:
    $e1 = "LD_PRELOAD" nocase
    $e2 = "LD_LIBRARY_PATH" nocase
    $e3 = "/etc/environment" nocase
    $e4 = "/etc/profile" nocase
    $e5 = ".bashrc" nocase
    $e6 = ".profile" nocase
    $e7 = "export " nocase
  condition:
    any of ($e1,$e2) and 1 of ($e3,$e4,$e5,$e6,$e7)
}

rule SH_Git_Hook_Persistence {
  meta:
    public_label = "git_hook_persistence"
    reason_key = "git_hook_persistence_pattern"
    reason_text = "git hook surfaces appear to be used for persistent or automatic execution"
  strings:
    $g1 = ".git/hooks/" nocase
    $g2 = "pre-commit" nocase
    $g3 = "post-checkout" nocase
    $g4 = "post-merge" nocase
    $g5 = "#!/bin/sh" nocase
    $g6 = "#!/bin/bash" nocase
  condition:
    $g1 and 1 of ($g2,$g3,$g4) and 1 of ($g5,$g6)
}
