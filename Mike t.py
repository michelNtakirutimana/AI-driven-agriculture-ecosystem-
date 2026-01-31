izina=input("andka izina ryawe: ")
ibiro=int(input("uyumusi winjijeibiro bingahe? "))
inyongera=ibiro*0.2
umusaruro_vubaha=ibiro+inyongera
print(f"muraho  {izina}, ejo biteganyijwe ko umusaruro wawe uzaba ari ibiro {umusaruro_vubaha} kg")
if ibiro <100:
      print("inama :  umusaruro wawe urihasi .koresha ifumbire yongera imbaraga")
elif  ibiro <= 200: 
      print("inama : umusaruro wawe urashimishije ! tangira gushaka isoko ryokugurishamo") 
else:
     print ("ujye wibanda kunamazabajyanama bubuhinzi")
print("-" * 20)