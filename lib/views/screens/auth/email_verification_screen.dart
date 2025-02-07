import 'dart:async';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import '../../../constants.dart';
import '../../../controllers/auth_controller.dart';

class EmailVerificationScreen extends StatefulWidget {
  const EmailVerificationScreen({super.key});

  @override
  State<EmailVerificationScreen> createState() => _EmailVerificationScreenState();
}

class _EmailVerificationScreenState extends State<EmailVerificationScreen> {
  final AuthController _authController = Get.find<AuthController>();
  Timer? _timer;
  RxBool isEmailVerified = false.obs;
  RxInt timeLeft = 60.obs;

  @override
  void initState() {
    super.initState();
    // Check verification status periodically
    _timer = Timer.periodic(
      const Duration(seconds: 3),
      (_) => _checkEmailVerified(),
    );
    // Timer for resend cooldown
    _startResendTimer();
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  Future<void> _checkEmailVerified() async {
    if (await _authController.checkEmailVerified()) {
      _timer?.cancel();
      Get.offAllNamed('/home');
    }
  }

  void _startResendTimer() {
    Timer.periodic(
      const Duration(seconds: 1),
      (Timer timer) {
        if (timeLeft.value > 0) {
          timeLeft.value--;
        } else {
          timer.cancel();
        }
      },
    );
  }

  Future<void> _resendVerification() async {
    if (timeLeft.value > 0) return;

    await _authController.sendEmailVerification();
    timeLeft.value = 60;
    _startResendTimer();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back, color: AppColors.white),
          onPressed: () => Get.back(),
        ),
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(AppTheme.spacing_lg),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(
                Icons.mark_email_unread_outlined,
                size: 80,
                color: AppColors.accent,
              ),
              const SizedBox(height: AppTheme.spacing_xl),
              Text(
                'Verify your email',
                style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                  color: AppColors.white,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: AppTheme.spacing_md),
              Text(
                'We sent a verification email to\n${_authController.user?.email}',
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                  color: AppColors.white.withOpacity(0.7),
                ),
              ),
              const SizedBox(height: AppTheme.spacing_xl),
              Obx(() => ElevatedButton(
                style: AppTheme.primaryButton,
                onPressed: timeLeft.value > 0 ? null : _resendVerification,
                child: Text(
                  timeLeft.value > 0
                      ? 'Resend in ${timeLeft.value}s'
                      : 'Resend Email',
                  style: const TextStyle(
                    fontSize: AppTheme.fontSize_md,
                    fontWeight: FontWeight.bold,
                    color: AppColors.black,
                  ),
                ),
              )),
              const SizedBox(height: AppTheme.spacing_lg),
              TextButton(
                onPressed: () async {
                  await _authController.signOut();
                  Get.offAllNamed('/login');
                },
                child: const Text(
                  'Change Email',
                  style: TextStyle(
                    color: AppColors.accent,
                    fontSize: AppTheme.fontSize_md,
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
