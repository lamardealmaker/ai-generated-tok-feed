import 'package:flutter/material.dart';
import 'package:get/get.dart';
import '../../../constants.dart';
import '../../../controllers/auth_controller.dart';
import 'signup_steps_screen.dart';

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();
  final _authController = Get.find<AuthController>();
  bool _isLogin = true;

  @override
  void dispose() {
    _emailController.dispose();
    _passwordController.dispose();
    super.dispose();
  }

  void _handleSubmit() async {
    final email = _emailController.text.trim();
    final password = _passwordController.text.trim();

    if (email.isEmpty || password.isEmpty) {
      Get.snackbar(
        'Error',
        'Please fill in all fields',
        snackPosition: SnackPosition.BOTTOM,
      );
      return;
    }

    if (_isLogin) {
      final result = await _authController.login(
        email: email,
        password: password,
      );

      if (result != 'success') {
        Get.snackbar(
          'Error',
          result,
          snackPosition: SnackPosition.BOTTOM,
        );
      }
    } else {
      // Navigate to signup steps
      Get.to(() => SignupStepsScreen(
        email: email,
        password: password,
      ));
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(32.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Text(
                _isLogin ? 'Welcome Back!' : 'Create Account',
                style: const TextStyle(
                  fontSize: 32,
                  fontWeight: FontWeight.bold,
                  color: AppColors.white,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 48),
              TextField(
                controller: _emailController,
                decoration: const InputDecoration(
                  hintText: 'Email',
                  prefixIcon: Icon(Icons.email),
                ),
                keyboardType: TextInputType.emailAddress,
              ),
              const SizedBox(height: 16),
              TextField(
                controller: _passwordController,
                decoration: const InputDecoration(
                  hintText: 'Password',
                  prefixIcon: Icon(Icons.lock),
                ),
                obscureText: true,
              ),
              const SizedBox(height: 24),
              Obx(() => _authController.isLoading.value
                  ? const Center(child: CircularProgressIndicator())
                  : ElevatedButton(
                      style: AppTheme.primaryButton,
                      onPressed: _handleSubmit,
                      child: Text(
                        _isLogin ? 'Log In' : 'Next',
                        style: const TextStyle(
                          fontSize: AppTheme.fontSize_md,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    )),
              const SizedBox(height: AppTheme.spacing_md),
              TextButton(
                style: TextButton.styleFrom(
                  foregroundColor: AppColors.white,
                ),
                onPressed: () {
                  setState(() => _isLogin = !_isLogin);
                },
                child: Text(
                  _isLogin
                      ? 'Don\'t have an account? Sign Up'
                      : 'Already have an account? Log In',
                  style: const TextStyle(
                    fontSize: AppTheme.fontSize_sm,
                  ),
                ),
              ),
              if (_isLogin) ...[
                const SizedBox(height: 8),
                TextButton(
                  onPressed: () {
                    // TODO: Implement forgot password
                    Get.snackbar(
                      'Coming Soon',
                      'Password reset will be available soon!',
                      snackPosition: SnackPosition.BOTTOM,
                    );
                  },
                  child: const Text(
                    'Forgot Password?',
                    style: TextStyle(color: AppColors.accent),
                  ),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}
