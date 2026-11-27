from notification_env import NotificationEnv

env = NotificationEnv(p_notif=0.5)
s, _ = env.reset()
total = 0
done = False

while not done:
    s, r, term, trunc, _ = env.step(1) # always press notification
    total += r
    done = term or trunc

print("steps:", env._t, "| total reward:", total)