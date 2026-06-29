import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


class MockServices(Node):

    def __init__(self):
        super().__init__('mock_services')
        self.create_service(Trigger, '/emergency_stop', self._estop_cb)
        self.create_service(Trigger, '/recover', self._recover_cb)
        self.get_logger().info('Mock services ready')

    def _estop_cb(self, req, res):
        self.get_logger().warn('>>> /emergency_stop CALLED <<<')
        res.success = True
        res.message = 'estop ok'
        return res

    def _recover_cb(self, req, res):
        self.get_logger().info('>>> /recover CALLED <<<')
        res.success = True
        res.message = 'recover ok'
        return res


def main():
    rclpy.init()
    node = MockServices()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
