import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from bboxes_ex_msgs.msg import BoundingBoxes
import math
import matplotlib.pyplot as plt
from rrt import RRT
import time 


def euler_from_quaternion(x, y, z, w):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

class RobotNavigation(Node):
    def __init__(self):
        super().__init__('robot_navigation')
        self.pub = self.create_publisher(Twist,'cmd_vel', 10)
        self.sub = self.create_subscription(Odometry,'odom', self.callback_odom, 10)
        self.sub_box = self.create_subscription( BoundingBoxes, "yolov7_detection/bounding_boxes",self.callback, 10)

        # Initial position 
        self.position_ = Point()
        self.yaw_ = 0
        # machine state
        self.state_ = 0
        self.Path = [[13.0,13.0]]
        
        # goal
        x_st , y_st = self.Path[0]
        self.desired_position_ = Point()
        self.desired_position_.x = float(x_st)
        self.desired_position_.y = float(y_st)
        self.desired_position_.z = 0.0
        # parameters
        self.yaw_precision_ = math.pi / 90 # +/- 2 degree allowed
        self.dist_precision_ = 0.01
        self.idex_path = 0

        # Path points
        self.start = (0.0,0.0)
        self.goal = (13.0, 13.0)
        self.obstacleList = []

        self.detect_object = None
        self.ob_x , self.ob_y = 0.0 ,0.0 
        # state for sub 
        self.once_sub = True

    def callback(self, msg_bbox):
        boxes = msg_bbox.bounding_boxes
        for bbox in boxes:
            print(bbox)
            if bbox.xmin > 200 and self.once_sub:
                self.detect_object =  bbox.class_id
                self.distance = bbox.center_dist

                            
    def fix_yaw(self):
        desired_yaw = math.atan2(self.desired_position_.y - self.position_.y, self.desired_position_.x - self.position_.x)
        err_yaw = desired_yaw - self.yaw_

        twist_msg = Twist()
        if math.fabs(err_yaw) > self.yaw_precision_:
            twist_msg.angular.z = 0.5 if err_yaw > 0 else -0.5
        
        self.pub.publish(twist_msg)
        
        # state change conditions
        if math.fabs(err_yaw) <= self.yaw_precision_:
            #print('Yaw error: [%s]' % err_yaw)
            self.change_state(1)

    def go_straight_ahead(self):

        desired_yaw = math.atan2(self.desired_position_.y - self.position_.y, self.desired_position_.x - self.position_.x)
        err_yaw = desired_yaw - self.yaw_
        err_pos = math.sqrt(pow(self.desired_position_.y - self.position_.y, 2) + pow(self.desired_position_.x - self.position_.x, 2))
        
        if err_pos > self.dist_precision_:
            twist_msg = Twist()
            twist_msg.linear.x = 0.5
            self.pub.publish(twist_msg)
        else:
            print('Position error: [%s]' % err_pos)
            self.change_state(2)
        
        # state change conditions
        if math.fabs(err_yaw) > self.yaw_precision_:
            print( 'Yaw error: [%s]' % err_yaw)
            self.change_state(0)

    def done(self):
        twist_msg = Twist()
        twist_msg.linear.x = 0.0
        twist_msg.angular.z = 0.0
        self.pub.publish(twist_msg)


    def detected_new_obstacle(self):
        if  self.position_.x+1 > self.ob_x and self.position_.y > self.ob_y+1:
            self.once_sub = True 
        else:
            pass
    def change_state(self, state):
        self.state_ = state
        #print('State changed to [%s]' % self.state_)

    def navigate_by_path(self):
        if int(len(self.Path)) > self.idex_path:
            print((self.Path))
            x_goal, y_goal = self.Path[self.idex_path]
            print(x_goal, y_goal )
            # Get the next position from the path list
            self.desired_position_.x = float(x_goal)
            self.desired_position_.y = float(y_goal) 
            self.change_state(0)  # Change state to fix_yaw for the new position
            self.idex_path +=1
        else:
            pass 

    def callback_odom(self, msg):
        # position
        self.position_ = msg.pose.pose.position
        self.start = [self.position_.x, self.position_.y]
        # yaw
        x = msg.pose.pose.orientation.x
        y = msg.pose.pose.orientation.y
        z = msg.pose.pose.orientation.z
        w = msg.pose.pose.orientation.w
        roll_x, pitch_y, yaw_z = euler_from_quaternion(x,y,z,w)
        self.yaw_ = yaw_z

        print(self.Path)

        self.obstacle()

        if self.state_ == 0:
            self.fix_yaw()
            self.detected_new_obstacle
            #print('Adjust Yaw')
        elif self.state_ == 1:
            self.go_straight_ahead()
            #print('Go Straight')
        elif self.state_ == 2:
            #print('Reach Goal')
            self.done()
            self.navigate_by_path()

        else:
            rclpy.get_logger().error('Unknown state!')
            pass

    def get_object_pose(self):
        (x1,y1) = self.start 
        (x2,y2) = self.goal 
        deta = math.tanh((x2-x1)/(y2-y1))
        x_object = self.position_.x + (self.distance * math.cos(deta))
        y_object = self.position_.y + (self.distance* math.sin(deta))
        return x_object, y_object

    def obstacle(self):
        if self.detect_object != None:
            
            if self.distance < 3.0 and self.once_sub : 
                self.once_sub = False 
                self.done()
                self.ob_x, self.ob_y = self.get_object_pose()
                # ====Search Path with RRT====
                obstacleList = [(self.ob_x, self.ob_y, 0.5)]
                # Set Initial parameters
                rrt = RRT(
                    [self.position_.x ,self.position_.y ],
                    self.goal,
                    rand_area=[-2, 15],
                    obstacle_list=obstacleList,
                    play_area=[0, 15, 0, 15],
                    robot_radius=0.8
                    )
                path = rrt.planning(animation=True)
                print(path)

                if path is None:
                    print("Cannot find path")
                else:
                    print("found path!!")
                    # Draw final path
                    if True:
                        rrt.draw_graph()
                        plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
                        plt.grid(True)
                        plt.pause(0.01)  # Need for Mac
                        plt.show()
                path.reverse()
                self.Path = path
                self.navigate_by_path()

            else: 
                print(" None Nearest Obstacle ")  
        else: 
            print(' Object is None ')

def main(args=None):
    rclpy.init(args=args)
    robot_navigation = RobotNavigation()
    rclpy.spin(robot_navigation)
    robot_navigation.destroy_node()
    rclpy.shutdown()
if __name__ == '__main__':
    main()

