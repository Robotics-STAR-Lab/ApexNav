from vlm.itm.clipitm import CLIPITMClient

itmclient = CLIPITMClient(port=12182)

def get_itm_message(rgb_image, label, return_stats=False):
    txt = f"Is there a {label} in the image?"
    response = itmclient.infer(rgb_image, txt)
    cosine = float(response["response"])
    itm_score = float(response["itm score"])
    if return_stats:
        return cosine, itm_score, response.get("timing", {})
    return cosine, itm_score

def get_itm_message_cosine(rgb_image, label, room, return_stats=False):
    if room != "everywhere":
        txt = f"Seems like there is a {room} or a {label} ahead?"
    else:
        txt = f"Seems like there is a {label} ahead?"
    response = itmclient.infer(rgb_image, txt)
    cosine = float(response["response"])
    if return_stats:
        return cosine, response.get("timing", {})
    return cosine
