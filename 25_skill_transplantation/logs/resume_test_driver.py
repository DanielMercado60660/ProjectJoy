import subprocess, sys, os, signal
S=sys.argv[1]; py='/Users/danielmercado/ProjectJoy/.venv/bin/python'
env=dict(os.environ, OMP_NUM_THREADS='1')
p=subprocess.Popen([py, S+'/rt.py', S+'/rt_ck'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env)
for line in p.stdout:
    print('RUN1', line.strip())
    if ' 40960 steps' in line:   # update 10 -> ckpt saved at update 10
        os.kill(p.pid, signal.SIGKILL); print('KILLED pid', p.pid); break
p.wait()
out=subprocess.run([py, S+'/rt.py', S+'/rt_ck'], capture_output=True, text=True, env=env).stdout
print('\n'.join('RUN2 '+l for l in out.splitlines() if 'resumed' in l or 'logged' in l or ' 4[59]' in l or '163840' in l))
