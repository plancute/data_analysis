# 많은 예제들이 tensorflow 1.x 버전을 기준으로 작성되어 tensorflow 2.0 최신버전으로 설치돼 위와같은 에러 등 제대로 실행이 안되는 문제가 있을것으로 생각됩니다
# tensorflow에서는 1.x 버전을 import 할 수 있도록 지원합니다 위와같이 tensorflow를 import하고 실행시키면 됩니다.
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

data=[[2,81],[4,93],[6,91],[8,97]]
x_data=[x_row[0] for x_row in data]
y_data=[y_row[1] for y_row in data]

# 기울기 a와 절편 b의 값을 임의로 정한다.
# 단 기울기의 범위는 0~10 사이이며, y절편은 0~100 사이에서 변하게 한다.
# AttributeError: module 'tensorflow' has no attribute 'random_uniform'
# 텐서플로우 2.0으로 오면서 random.uniform()으로 바뀐듯 하다.
a= tf.Variable(tf.random.uniform([1],0,10,dtype=tf.float64, seed=0))
b= tf.Variable(tf.random.uniform([1],0,100,dtype=tf.float64, seed=0))


# y에 # y에 대한 일차 방정식 ax+b의 식을 세운다.
y = a * x_data + b

# 텐서플로 RMSE 함수
rmse = tf.sqrt(tf.reduce_mean(tf.square(y - y_data)))

# 학습률 값
learning_rate = 0.1

# RMSE 값을 최소로 하는 값 찾기
gradient_descent = tf.train.GradientDescentOptimizer(learning_rate).minimize(rmse)

# 텐서플로를 이용한 학습
with tf.Session() as sess:
  # 변수 초기화
  sess.run(tf.global_variables_initializer())
  # 2001번 실행(0번째를 포함하므로)
  for step in range(2001):
    sess.run(gradient_descent)
    # 100번마다 결과 출력
    if step % 100 == 0:
      print("Epoch : %.f RMSE : %.04f, 기울기 a = %.4f, y절편 b=%.4f" % (step, sess.run(rmse),sess.run(a),sess.run(b)))
