from rest_framework.views import APIView
from rest_framework.response import Response
import requests
from user_app.models import ImageHash,CustomUser
from user_app.api.serializer import ImageHashSerializer,UserRegistrationSerializer
import json

class TrainView(APIView):
    def get(self, request, *args, **kwargs):
        try:
            data  = ImageHash.objects.all()#.values('base64_1','base64_2','base64_3','base64_4','base64_5','customerId')
            serializer = ImageHashSerializer(instance = data, many=True)
            if serializer:
                serialized_data = serializer.data
                json_serialized =json.dumps(serialized_data)
                json_data = {"mode":"train",
                             "data": json_serialized
                             }
                parsed_data = json.loads(json_data['data'])
                json_data['data'] = parsed_data
                
                if len(json_data['data']) == 0:
                    return Response({"message": "There is no data for train!"},status=204)
                for item in json_data['data']:
                    # del item['id']
                    del item['vdid']
                    del item['vd_path']
                    item['customer_id'] = item.pop('customerId')
                json_string = json.dumps(json_data)
                json_string1 = json.loads(json_string)
                header = {"Content-Type": "application/json"}
                
                response = requests.post('http://65.2.83.6:6000',json=json_string1)  # Replace with the actual API URL
                response_data = response.json()
                if response_data['status'] == "complete":
                    response_data_list = response_data["vdb_id"]

                    for item in response_data_list:
                        print(item['cust_id'])
                        try: 
                            customUser_instance = CustomUser.objects.get(customer_id = item['cust_id'])
                            updated_data = ImageHash.objects.get(customerId = customUser_instance)
                            print(updated_data,"updated data")
                            item['customerId'] = item.pop('cust_id')
                            item['vdid'] = item.pop('vdb_id')
                            item['vd_path'] = item.pop('vdb_path')
                            serializer = ImageHashSerializer(updated_data, data=item, partial=True)
                            print(item,"item")
                            serializer.is_valid(raise_exception=True)

                            if serializer.is_valid():
                                print("here")
                                print(serializer.validated_data)

                                print("serializer_valid")
                                serializer.save()

                        except ImageHash.DoesNotExist:
                            return Response({"message": "User does not exist"}, status=404)

                return Response({'data' : response_data})

        except requests.exceptions.RequestException as e:
            return Response({'error': str(e)}, status=500)
class TestView(APIView):
    def post(self, request, *args, **kwargs):
        data = request.data
        api_url = "http://52.66.166.159:6000"
        response = requests.post(api_url,json = data)
        # return Response(response.json())
        response_data = response.json()
        if response_data['verified']== True:
            try:
                print(response_data["vdb_id"])
                hash_table = ImageHash.objects.get(vdid= response_data["vdb_id"])
                user_details  = CustomUser.objects.get(customer_id = hash_table)
                serializer  = UserRegistrationSerializer(instance = user_details)
                serialized_data = serializer.data
                if "images" in serialized_data:
                    serialized_data.pop('images')

                return Response(serialized_data)
                # else:
                #     print(serializer.errors)
            except ImageHash.DoesNotExist or CustomUser.DoesNotExist:
                return Response({"message": "User does not exist"}, status=404)
        else:
            response={
                "message":"user verification Failed!"
            }
            return Response(response,status=400)
