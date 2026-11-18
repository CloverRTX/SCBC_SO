def train_record_log(loop, clock, str = ''):
    with open(r"Log\training_log.txt", "a+", encoding='utf-8') as f:
        f.write("========================training_log========================\n")
        f.write("————————————start training————————————\n")
        f.write(str + "\n")
        f.write(f"The loop is {loop}\n")
        f.write(f"start time is: {clock.start}\n")
        f.write(f"end time is: {clock.end}\n")
        f.write(f"cost : {clock.timeCost_str()}\n")


def verify_record_log(loop, clock, str = ''):
    with open(r"Log\verify_log.txt", "a+", encoding='utf-8') as f:
        f.write("========================verify_log========================\n")
        f.write("————————————start verify————————————\n")
        f.write(str + "\n")
        f.write(f"The loop is {loop}\n")
        f.write(f"start time is: {clock.start}\n")
        f.write(f"end time is: {clock.end}\n")
        f.write(f"cost : {clock.timeCost_str()}\n")

def info_log(ep, total_Xu_loss, total_X_loss, total_kI_ME_loss, gama, ME_upper, Tag = "Train"):
    with open(r"Log\info_log.txt", "a+", encoding='utf-8') as f:
        f.write(f"====================[{Tag}] The ep is {ep}====================\n")
        f.write(f"total_Xu_loss : {total_Xu_loss}\n")
        f.write(f"total_X_loss : {total_X_loss}\n")
        f.write(f"total_kI_ME_loss : {total_kI_ME_loss}\n")
        f.write(f"gama : {gama}\n")
        f.write(f"ME_upper : {ME_upper}\n")
        f.write(f"Psafe : {-1}\n")

def info_log_str(str):
    with open(r"Log\info_log.txt", "a+", encoding='utf-8') as f:
        f.write(f"{str}\n")
